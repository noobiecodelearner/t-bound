"""[sdk/recommender). — recommendation fetcher."""

from sdk.exceptions import TBoundAPIError


class Recommendation:
    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, v)


def get_recommendation(api_url: str, api_key: str, target_accuracy: float) -> Recommendation:
    try:
        import httpx
        r = httpx.get(
            f"{api_url}/v1/recommend",
            params={"target_accuracy": target_accuracy},
            headers={"X-TBound-Key": api_key},
            timeout=15,
        )
        r.raise_for_status()
        return Recommendation(r.json())
    except Exception as e:
        raise TBoundAPIError(f"Failed to get recommendation: {e}") from e
