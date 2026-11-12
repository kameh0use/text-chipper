use std::borrow::Cow;
use unicode_normalization::{
    char::canonical_combining_class, is_nfc_quick, is_nfkc_quick, IsNormalized,
    UnicodeNormalization,
};

use super::{chunk_types::Chunk, ChipError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    None,
    Nfc,
    Nfkc,
}

impl Normalization {
    pub fn parse(s: &str) -> Result<Self, ChipError> {
        match s.to_ascii_lowercase().as_str() {
            "none" | "" => Ok(Normalization::None),
            "nfc" => Ok(Normalization::Nfc),
            "nfkc" => Ok(Normalization::Nfkc),
            other => Err(ChipError::InvalidConfig(format!(
                "normalize must be one of 'nfc', 'nfkc', 'none' (got '{other}')"
            ))),
        }
    }

    pub fn apply<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // ASCII is invariant under NFC/NFKC.
        if text.is_ascii() {
            return Cow::Borrowed(text);
        }
        match self {
            Normalization::None => Cow::Borrowed(text),
            Normalization::Nfc => {
                if matches!(is_nfc_quick(text.chars()), IsNormalized::Yes) {
                    return Cow::Borrowed(text);
                }
                let n: String = text.nfc().collect();
                if n == text {
                    Cow::Borrowed(text)
                } else {
                    Cow::Owned(n)
                }
            }
            Normalization::Nfkc => {
                if matches!(is_nfkc_quick(text.chars()), IsNormalized::Yes) {
                    return Cow::Borrowed(text);
                }
                let n: String = text.nfkc().collect();
                if n == text {
                    Cow::Borrowed(text)
                } else {
                    Cow::Owned(n)
                }
            }
        }
    }
}

/// Running totals (normalized/original bytes and chars) before the text being normalized.
#[derive(Clone, Copy, Debug, Default)]
pub struct NormBase {
    pub nb: usize,
    pub ob: usize,
    pub nc: usize,
    pub oc: usize,
}

/// One segment that normalization changed. `[norm_start, norm_end, orig_start, orig_end]`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NormSeg {
    pub(crate) b: [usize; 4],
    pub(crate) c: [usize; 4],
}

/// Position of a chunk in the ORIGINAL (un-normalized) text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrigSpan {
    pub start_char: usize,
    pub end_char: usize,
    pub start_byte: usize,
    pub end_byte: usize,
}

/// Piecewise-linear map. Only segments that normalization actually changed are stored; between
/// them offsets differ by a constant. A position strictly inside a changed segment (e.g. between
/// the two letters of an NFKC-expanded ligature) maps to the segment start (chunk start) or the
/// segment end (chunk end), so the original span always covers the whole chunk.
#[derive(Debug, Default)]
pub struct NormMap {
    pub(crate) segs: Vec<NormSeg>,
    base_db: isize,
    base_dc: isize,
}

fn map_with(
    segs: &[NormSeg],
    base: isize,
    x: usize,
    ceil: bool,
    sel: fn(&NormSeg) -> [usize; 4],
) -> usize {
    let idx = segs.partition_point(|s| sel(s)[1] <= x);
    if idx < segs.len() {
        let [ns, _ne, os, oe] = sel(&segs[idx]);
        if ns < x {
            return if ceil { oe } else { os };
        }
    }
    let delta = if idx == 0 {
        base
    } else {
        let [_, ne, _, oe] = sel(&segs[idx - 1]);
        oe as isize - ne as isize
    };
    (x as isize + delta).max(0) as usize
}

impl NormMap {
    pub fn map_byte(&self, x: usize, ceil: bool) -> usize {
        map_with(&self.segs, self.base_db, x, ceil, |s| s.b)
    }

    pub fn map_char(&self, x: usize, ceil: bool) -> usize {
        map_with(&self.segs, self.base_dc, x, ceil, |s| s.c)
    }

    pub fn span(&self, c: &Chunk) -> OrigSpan {
        OrigSpan {
            start_char: self.map_char(c.start_char, false),
            end_char: self.map_char(c.end_char, true),
            start_byte: self.map_byte(c.start_byte, false),
            end_byte: self.map_byte(c.end_byte, true),
        }
    }

    /// Streaming: forget segments that lie entirely before `nb_pos`/`nc_pos` (chunk starts never
    /// move backwards), keeping memory bounded on huge files.
    pub fn prune(&mut self, nb_pos: usize, nc_pos: usize) {
        if self.segs.len() < 4096 {
            return;
        }
        let k = self
            .segs
            .partition_point(|s| s.b[1] <= nb_pos && s.c[1] <= nc_pos);
        if k == 0 {
            return;
        }
        let last = self.segs[k - 1];
        self.base_db = last.b[3] as isize - last.b[1] as isize;
        self.base_dc = last.c[3] as isize - last.c[1] as isize;
        self.segs.drain(..k);
    }
}

impl Normalization {
    /// A position before `ch` is a safe place to cut the text into independently normalizable
    /// pieces: `ch` is a starter (ccc = 0) that never composes with the character before it.
    /// ASCII always qualifies. (Compat-jamo mixed with conjoining jamo under NFKC is the one
    /// exotic case this does not model.)
    fn is_norm_boundary(&self, ch: char) -> bool {
        if ch.is_ascii() {
            return true;
        }
        if canonical_combining_class(ch) != 0 {
            return false;
        }
        match self {
            Normalization::None => true,
            Normalization::Nfc => !matches!(is_nfc_quick(std::iter::once(ch)), IsNormalized::Maybe),
            Normalization::Nfkc => {
                !matches!(is_nfkc_quick(std::iter::once(ch)), IsNormalized::Maybe)
            }
        }
    }

    fn flush_seg(
        &self,
        seg: &str,
        out: &mut String,
        st: &mut [usize; 4], // local [orig_byte, orig_char, norm_byte, norm_char]
        segs: &mut Vec<NormSeg>,
        base: &NormBase,
    ) {
        let before = out.len();
        if seg.is_ascii() {
            out.push_str(seg);
        } else {
            match self {
                Normalization::Nfc => out.extend(seg.nfc()),
                Normalization::Nfkc => out.extend(seg.nfkc()),
                Normalization::None => out.push_str(seg),
            }
        }
        let n_len = out.len() - before;
        let o_chars = seg.chars().count();
        let changed = &out[before..] != seg;
        let n_chars = if changed {
            out[before..].chars().count()
        } else {
            o_chars
        };
        if changed {
            segs.push(NormSeg {
                b: [
                    base.nb + st[2],
                    base.nb + st[2] + n_len,
                    base.ob + st[0],
                    base.ob + st[0] + seg.len(),
                ],
                c: [
                    base.nc + st[3],
                    base.nc + st[3] + n_chars,
                    base.oc + st[1],
                    base.oc + st[1] + o_chars,
                ],
            });
        }
        st[0] += seg.len();
        st[1] += o_chars;
        st[2] += n_len;
        st[3] += n_chars;
    }

    /// Like `apply`, but records how normalized offsets map back to the original text.
    /// `base` = totals consumed so far (zeros for a whole text, running sums when streaming).
    /// Unchanged text costs nothing: it is returned borrowed and nothing is recorded.
    pub fn apply_mapped<'a>(
        &self,
        text: &'a str,
        base: NormBase,
        map: &mut NormMap,
    ) -> Cow<'a, str> {
        if text.is_ascii() || *self == Normalization::None {
            return Cow::Borrowed(text);
        }
        let quick = match self {
            Normalization::Nfc => is_nfc_quick(text.chars()),
            Normalization::Nfkc => is_nfkc_quick(text.chars()),
            Normalization::None => IsNormalized::Yes,
        };
        if matches!(quick, IsNormalized::Yes) {
            return Cow::Borrowed(text);
        }

        let mut out = String::with_capacity(text.len() + 16);
        let mut segs: Vec<NormSeg> = Vec::new();
        let mut st = [0usize; 4];
        let mut seg_start = 0usize;
        for (i, ch) in text.char_indices() {
            if i > 0 && self.is_norm_boundary(ch) {
                self.flush_seg(&text[seg_start..i], &mut out, &mut st, &mut segs, &base);
                seg_start = i;
            }
        }
        self.flush_seg(&text[seg_start..], &mut out, &mut st, &mut segs, &base);

        if segs.is_empty() {
            return Cow::Borrowed(text);
        }
        map.segs.extend(segs);
        Cow::Owned(out)
    }
}
