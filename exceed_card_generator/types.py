from collections import namedtuple

FontFamily = namedtuple("FontFamily", ["regular", "bold", "italic"])
Keyword = namedtuple("Keyword", ["pattern", "begin", "end"])
TextChunk = namedtuple("TextChunk", ["text"])
KwargsChunk = namedtuple("ColorChunk", ["kwargs"])
Chunk = TextChunk | KwargsChunk
