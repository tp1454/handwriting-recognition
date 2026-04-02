"""Tests for oversized box splitting and per-group similarity scoring."""

import numpy as np
import pytest
from src.data.processing import split_big_boxes_and_score_similarity


def test_split_big_box_and_scores_are_bounded():
    """Oversized horizontal boxes should be split and scored in [0, 100]."""
    image = np.zeros((20, 40), dtype=np.uint8)
    image[4:16, 2:8] = 255
    image[4:16, 10:16] = 255

    detect_output = {
        "horizontal_list": [
            [0, 20, 0, 20],
            [24, 32, 0, 20],
        ],
        "free_list": [],
    }

    result = split_big_boxes_and_score_similarity(
        image,
        detect_output,
        width_ratio=1.2,
        vector_size=16,
    )

    groups = result["horizontal_groups"]
    assert len(groups) == 2
    assert groups[0]["was_split"] is True
    assert len(groups[0]["items"]) == 2
    assert groups[0]["items"][0][
        "similarity_to_first"
    ] == pytest.approx(100.0)

    for group in groups:
        for item in group["items"]:
            assert 0.0 <= item["similarity_to_first"] <= 100.0


def test_first_split_is_reference_for_group_similarity():
    """Second split should be compared to the first split from the same source box."""
    image = np.zeros((24, 40), dtype=np.uint8)

    # Two intentionally similar shapes inside the oversized source box.
    image[5:19, 2:8] = 255
    image[5:19, 12:18] = 255

    detect_output = {
        "horizontal_list": [
            [0, 20, 0, 24],
            [22, 30, 0, 24],
        ],
        "free_list": [],
    }

    result = split_big_boxes_and_score_similarity(
        image,
        detect_output,
        width_ratio=1.1,
        vector_size=20,
    )

    first_group = result["horizontal_groups"][0]
    assert first_group["was_split"] is True
    assert len(first_group["items"]) == 2
    assert first_group["items"][1]["similarity_to_first"] > 95.0


def test_no_split_when_box_not_oversized():
    """Boxes under the relative width rule should remain single segments."""
    image = np.zeros((20, 30), dtype=np.uint8)

    detect_output = {
        "horizontal_list": [
            [0, 10, 0, 20],
            [12, 23, 0, 20],
        ],
        "free_list": [],
    }

    result = split_big_boxes_and_score_similarity(
        image,
        detect_output,
        width_ratio=2.0,
        vector_size=12,
    )

    assert len(result["horizontal_groups"]) == 2
    assert len(result["horizontal_list"]) == 2
    assert all(
        group["was_split"] is False
        and len(group["items"]) == 1
        and group["items"][0]["similarity_to_first"]
        == pytest.approx(100.0)
        for group in result["horizontal_groups"]
    )


def test_ignores_invalid_boxes_and_preserves_free_list():
    """Invalid horizontal entries should be skipped without dropping free boxes."""
    image = np.zeros((8, 8), dtype=np.uint8)

    detect_output = {
        "horizontal_list": [
            [],
            ["x"],
            [0, 0, 0, 5],
            [2, 6, 1, 7],
        ],
        "free_list": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
    }

    result = split_big_boxes_and_score_similarity(
        image,
        detect_output,
        width_ratio=1.5,
    )

    assert len(result["horizontal_groups"]) == 1
    assert result["horizontal_groups"][0]["source_box"] == [
        2,
        6,
        1,
        7,
    ]
    assert result["free_list"] == detect_output["free_list"]
