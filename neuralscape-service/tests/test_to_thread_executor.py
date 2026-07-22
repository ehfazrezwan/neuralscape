"""Tests for asyncio.to_thread executor configuration."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_configure_to_thread_executor_sets_custom_size():
    """When to_thread_max_workers>0, sets the loop's default executor to that size."""
    from config import settings
    from main import _configure_to_thread_executor

    loop = asyncio.get_running_loop()

    with (
        patch.object(settings, 'to_thread_max_workers', 24),
        patch.object(loop, 'set_default_executor', MagicMock()) as spy,
    ):
        result = _configure_to_thread_executor(loop)
        assert result == 24

        # Assert on the public call, not loop internals: exactly one
        # ThreadPoolExecutor of the configured size was installed.
        spy.assert_called_once()
        executor = spy.call_args.args[0]
        assert isinstance(executor, ThreadPoolExecutor)
        assert executor._max_workers == 24
        executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_configure_to_thread_executor_with_zero_returns_none():
    """When to_thread_max_workers=0 (default), leaves stdlib default and returns None."""
    from config import settings
    from main import _configure_to_thread_executor

    loop = asyncio.get_running_loop()

    with (
        patch.object(settings, 'to_thread_max_workers', 0),
        patch.object(loop, 'set_default_executor', MagicMock()) as spy,
    ):
        result = _configure_to_thread_executor(loop)
        assert result is None
        spy.assert_not_called()


@pytest.mark.asyncio
async def test_configure_to_thread_executor_with_none_returns_none():
    """When to_thread_max_workers is None, leaves stdlib default and returns None."""
    from config import settings
    from main import _configure_to_thread_executor

    loop = asyncio.get_running_loop()

    with (
        patch.object(settings, 'to_thread_max_workers', None),
        patch.object(loop, 'set_default_executor', MagicMock()) as spy,
    ):
        result = _configure_to_thread_executor(loop)
        assert result is None
        spy.assert_not_called()
