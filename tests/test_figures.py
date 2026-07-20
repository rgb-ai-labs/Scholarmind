
from scholarmind.agents.base import AgentResult
from scholarmind.agents.figures import (
    CAPTION_ONLY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    answer_about_figure,
)
from scholarmind.config import Settings
from scholarmind.retrieval.dense import DenseResult


class _CaptionOnlyClient:
    # Does NOT implement complete_with_image, mirroring a plain FakeLLMClient test double.
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0
        self.last_system_prompt = ""
        self.last_user_prompt = ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response


class _VisionCapableClient(_CaptionOnlyClient):
    def __init__(self, response: str, image_response: str) -> None:
        super().__init__(response)
        self.image_response = image_response
        self.image_call_count = 0
        self.last_image_path = None
        self.last_model = None

    def complete_with_image(
        self, system_prompt: str, user_prompt: str, image_path: str, model: str
    ) -> str:
        self.image_call_count += 1
        self.last_image_path = image_path
        self.last_model = model
        return self.image_response


def _figure(image_path: str | None) -> DenseResult:
    return DenseResult(
        paper_id="p1",
        title="A Paper",
        authors=["Ada Lovelace"],
        year=2020,
        venue=None,
        section="Figure 1: A chart.",
        page_start=3,
        page_end=3,
        chunk_index=5,
        text="Figure (page 3): Figure 1: A chart.",
        score=0.0,
        chunk_type="figure",
        image_path=image_path,
    )


def test_answer_about_figure_falls_back_to_caption_only_when_no_vision_model_configured(
    tmp_path,
):
    image_path = tmp_path / "fig.png"
    image_path.write_bytes(b"fake png")
    figure = _figure(str(image_path))

    client = _VisionCapableClient("should not be used", "described from image")
    settings = Settings(vision_model="")  # not configured

    result = answer_about_figure(figure, "What does the chart show?", client, settings)

    assert isinstance(result, AgentResult)
    assert client.image_call_count == 0
    assert client.call_count == 1
    assert client.last_system_prompt == CAPTION_ONLY_SYSTEM_PROMPT
    assert result.sources == [figure]
    assert result.sources_found == 1


def test_answer_about_figure_uses_vision_when_configured_and_image_exists(tmp_path):
    image_path = tmp_path / "fig.png"
    image_path.write_bytes(b"fake png")
    figure = _figure(str(image_path))

    client = _VisionCapableClient("caption-only answer", "vision answer")
    settings = Settings(vision_model="some-vision-model")

    result = answer_about_figure(figure, "What does the chart show?", client, settings)

    assert result.text == "vision answer"
    assert client.image_call_count == 1
    assert client.call_count == 0
    assert client.last_model == "some-vision-model"
    assert client.last_image_path == str(image_path)


def test_answer_about_figure_falls_back_when_image_file_missing(tmp_path):
    figure = _figure(str(tmp_path / "does_not_exist.png"))

    client = _VisionCapableClient("caption-only answer", "vision answer")
    settings = Settings(vision_model="some-vision-model")

    result = answer_about_figure(figure, "What does the chart show?", client, settings)

    assert result.text == "caption-only answer"
    assert client.image_call_count == 0


def test_answer_about_figure_falls_back_when_client_lacks_vision_support(tmp_path):
    image_path = tmp_path / "fig.png"
    image_path.write_bytes(b"fake png")
    figure = _figure(str(image_path))

    client = _CaptionOnlyClient("caption-only answer")
    settings = Settings(vision_model="some-vision-model")

    result = answer_about_figure(figure, "What does the chart show?", client, settings)

    assert result.text == "caption-only answer"
    assert client.last_system_prompt == CAPTION_ONLY_SYSTEM_PROMPT


def test_answer_about_figure_falls_back_when_vision_call_raises(tmp_path):
    image_path = tmp_path / "fig.png"
    image_path.write_bytes(b"fake png")
    figure = _figure(str(image_path))

    class _FailingVisionClient(_VisionCapableClient):
        def complete_with_image(self, *args, **kwargs):
            self.image_call_count += 1
            raise RuntimeError("vision API down")

    client = _FailingVisionClient("caption-only answer", "unused")
    settings = Settings(vision_model="some-vision-model")

    result = answer_about_figure(figure, "What does the chart show?", client, settings)

    assert result.text == "caption-only answer"
    assert client.image_call_count == 1
    assert client.call_count == 1


def test_answer_about_figure_vision_system_prompt_differs_from_caption_only():
    assert SYSTEM_PROMPT != CAPTION_ONLY_SYSTEM_PROMPT
