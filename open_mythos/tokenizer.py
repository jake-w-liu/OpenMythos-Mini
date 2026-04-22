DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


class ByteTokenizer:
    """
    Minimal byte-level tokenizer for local, low-resource experiments.

    Uses raw UTF-8 bytes as tokens, giving a fixed 256-token vocabulary with
    no external files or downloads. This keeps the training path self-contained
    for smoke tests, laptop runs, and tiny research loops.
    """

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(int(i) % 256 for i in token_ids).decode(
            "utf-8", errors="replace"
        )


class MythosTokenizer:
    """
    HuggingFace tokenizer wrapper for OpenMythos.

    Args:
        model_id (str): The HuggingFace model ID or path to use with AutoTokenizer.
            Defaults to "openai/gpt-oss-20b".

    Attributes:
        tokenizer: An instance of HuggingFace's AutoTokenizer.

    Example:
        >>> tok = MythosTokenizer()
        >>> ids = tok.encode("Hello world")
        >>> s = tok.decode(ids)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        """
        Initialize the MythosTokenizer.

        Args:
            model_id (str): HuggingFace model identifier or path to tokenizer files.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "MythosTokenizer requires the 'transformers' package. "
                "Install it or use ByteTokenizer for low-resource local experiments."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @property
    def vocab_size(self) -> int:
        """
        Return the size of the tokenizer vocabulary.

        Returns:
            int: The number of unique tokens in the tokenizer vocabulary.
        """
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        """
        Encode input text into a list of token IDs.

        Args:
            text (str): The input text string to tokenize.

        Returns:
            list[int]: List of integer token IDs representing the input text.
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a text string.

        Args:
            token_ids (list[int]): A list of integer token IDs to decode.

        Returns:
            str: Decoded string representation of the token IDs.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
