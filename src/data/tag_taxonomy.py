"""ElevenLabs v3 audio tag taxonomy for Evoxtral."""

EMOTION_TAGS = ["excited", "sad", "angry", "nervous", "calm", "frustrated"]
NONVERBAL_TAGS = ["laughs", "sighs", "gasps", "clears throat", "crying"]
DELIVERY_TAGS = ["whispers", "shouts", "stammers"]
PAUSE_TAGS = ["pause"]

ALL_BRACKET_TAGS = EMOTION_TAGS + NONVERBAL_TAGS + DELIVERY_TAGS + PAUSE_TAGS

# Slice definitions for balanced dataset
SLICE_CONFIG = {
    "plain": {"ratio": 0.25, "tag_density": 0, "description": "No tags, plain ASR"},
    "light": {"ratio": 0.25, "tag_density": (1, 2), "description": "1-2 tags per sample"},
    "moderate": {"ratio": 0.25, "tag_density": (3, 4), "description": "3-4 tags per sample"},
    "dense": {"ratio": 0.15, "tag_density": (5, 8), "description": "5+ tags per sample"},
    "edge": {"ratio": 0.10, "tag_density": (1, 6), "description": "Edge cases: ambiguous, boundary"},
}

DOMAINS = [
    "conversation", "monologue", "podcast", "presentation",
    "argument", "storytelling", "interview", "voicemail"
]

# Semantic groups for eval (tags within a group are considered equivalent)
TAG_SEMANTIC_GROUPS = {
    "laughter": ["laughs", "giggles", "chuckles"],
    "sadness": ["sad", "crying", "sorrowful"],
    "breathing": ["sighs", "gasps", "exhales"],
    "loud": ["shouts", "yells"],
    "quiet": ["whispers", "murmurs"],
}
