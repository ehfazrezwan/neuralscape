"""Tests for asyncio.to_thread executor configuration."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_configure_to_thread_executor_sets_custom_size():
    """When to_thread_max_workers>0, sets the loop's default executor to that size."""
    from config import settings
    from main import _configure_to_thread_executor

    loop = asyncio.get_running_loop()

    with patch.object(settings, 'to_thread_max_workers', 24):
        result = _configure_to_thread_executor(loop)
        assert result == 24

        # Verify the executor was set
        executor = loop._default_executor
        assert executor is not None
        assert isinstance(executor, ThreadPoolExecutor)
        assert executor._max_workers == 24


@pytest.mark.asyncio
async def test_configure_to_thread_executor_with_zero_returns_none():
    """When to_thread_max_workers=0 (default), leaves stdlib default and returns None."""
    from config import settings
    from main import _configure_to_thread_executor

    loop = asyncio.get_running_loop()
    original_executor = loop._default_executor

    with patch.object(settings, 'to_thread_max_workers', 0):
        result = _configure_to_thread_executor(loop)
        assert result is None
        # Executor should be unchanged
        assert loop._default_executor == original_executor


@pytest.mark.asyncio
async def test_configure_to_thread_executor_with_none_returns_none():
    """When to_thread_max_workers is None, leaves stdlib default and returns None."""
    from config import settings
    from main import _configure_to_thread_executor

    loop = asyncio.get_running_loop()
    original_executor = loop._default_executor

    with patch.object(settings, 'to_thread_max_workers', None):
        result = _configure_to_thread_executor(loop)
        assert result is None
        # Executor should be unchanged
        assert loop._default_executor == original_executor
