| Lane | Mode | Runs | Time (s) | MiB/s | Peak memory growth (MB) | CPU / wall | Chunks | Largest chunk (tokens) | Over limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single | text_chipper:stream | 3 | 25.1 | 20.5 | 56 | 1.0 | 333,939 | 512 | 0 | ok |
| single | text_chipper:whole | 3 | 35.6 | 14.4 | 4623 | 1.0 | 333,939 | 512 | 0 | ok |
| single | text_chipper:blocks | 3 | 28.0 | 18.3 | 140 | 1.0 | 291,902 | 512 | 0 | ok |
| single | tiktoken_window:blocks | 3 | 40.2 | 12.8 | 149 | 1.0 | 234,790 | 513 | 5 | ok |
| single | chonkie_token:blocks | 3 | 46.4 | 11.1 | 208 | 1.0 | 234,790 | 513 | 5 | ok |
| single | chonkie_recursive:blocks | 3 | 41.2 | 12.5 | 110 | 1.0 | 290,436 | 512 | 0 | ok |
| single | semantic_text_splitter:blocks | 3 | 151.9 | 3.4 | 87 | 1.0 | 288,147 | 512 | 0 | ok |
| single | semchunk:blocks | 3 | 110.8 | 4.6 | 3321 | 1.0 | 289,847 | 512 | 0 | ok |
| single | langchain:blocks | 3 | 105.8 | 4.9 | 98 | 1.0 | 292,611 | 513 | 10 | ok |
| single | tiktoken_window:whole | 3 | 43.2 | 11.9 | 7706 | 1.0 | 234,759 | 513 | 6 | ok |
| single | chonkie_token:whole | 3 | 73.0 | 7.1 | 9033 | 1.0 | 234,759 | 513 | 6 | ok |
| single | chonkie_recursive:whole | 3 | 47.2 | 10.9 | 4624 | 1.0 | 290,415 | 512 | 0 | ok |
| single | semantic_text_splitter:whole | 3 | 492.0 | 1.0 | 4621 | 1.0 | 288,127 | 512 | 0 | ok |
| single | semchunk:whole | 3 | 118.4 | 4.3 | 6603 | 1.0 | 289,827 | 512 | 0 | ok |
| single | langchain:whole | 3 | 116.7 | 4.4 | 4624 | 1.0 | 292,590 | 513 | 10 | ok |
