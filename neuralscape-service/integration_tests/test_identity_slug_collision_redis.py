"""First-login slug collisions against a real Redis (sec-07).

Two different emails can derive one slug (``a+b@x`` and ``a-b@x`` both give
``a-b-x``). The durable identity store must keep the first person's id and
give the second a hash-suffixed one, and both must resolve back identically
on re-login. Uses the service's own ``identity_store`` over the gate's Redis;
every field written is removed on cleanup.
"""

from __future__ import annotations

import asyncio


def test_second_identity_with_a_claimed_slug_gets_its_own_id(flags, stores, unique_ns):
    import identity_store
    from login_providers import provision_user_id, resolve_login_identity

    flags(redis_url=stores.redis_url, auth_identity_map="")

    domain = f"{unique_ns.user_id}.example.com"
    first_email, second_email = f"a-b@{domain}", f"a+b@{domain}"
    first_sub, second_sub = f"{unique_ns.user_id}-s1", f"{unique_ns.user_id}-s2"
    unique_ns.on_cleanup(
        lambda: asyncio.run(
            _cleanup(identity_store, (first_sub, first_email), (second_sub, second_email))
        )
    )

    async def scenario():
        uid1, linked1 = await resolve_login_identity(first_sub, first_email)
        assert linked1 is False
        await identity_store.link(uid1, sub=first_sub, email=first_email)

        uid2, linked2 = await resolve_login_identity(second_sub, second_email)
        assert linked2 is False
        assert uid2 != uid1 and uid2.startswith(uid1 + "-")  # same slug, hash suffix
        # /oauth/link action=new re-checks and is idempotent for the same identity
        assert await provision_user_id(uid1, second_sub, second_email) == uid2
        await identity_store.link(uid2, sub=second_sub, email=second_email)

        # both resolve to their own, distinct, stable ids on the next login
        assert await resolve_login_identity(first_sub, first_email) == (uid1, True)
        assert await resolve_login_identity(second_sub, second_email) == (uid2, True)
        return uid1, uid2

    uid1, uid2 = asyncio.run(scenario())
    assert uid1 != uid2


async def _cleanup(identity_store, *pairs):
    for sub, email in pairs:
        await identity_store.unlink(sub=sub, email=email)


def test_concurrent_first_logins_cannot_share_a_slug(flags, stores, unique_ns):
    """Two simultaneous /oauth/link provisionings on one slug: one plain, one suffixed."""

    import identity_store
    from login_providers import provision_user_id

    flags(redis_url=stores.redis_url, auth_identity_map="")
    domain = f"{unique_ns.user_id}.example.com"
    ids = [(f"{unique_ns.user_id}-c{i}", f"a{sep}b@{domain}") for i, sep in enumerate("-+")]
    slug = f"a-b-{domain}"
    unique_ns.on_cleanup(lambda: asyncio.run(_cleanup(identity_store, *ids)))

    async def provision(sub, email):
        uid = await provision_user_id(slug, sub, email, reserve=True)
        await identity_store.link(uid, sub=sub, email=email)
        return uid

    async def both():
        return await asyncio.gather(*(provision(sub, email) for sub, email in ids))

    got = asyncio.run(both())
    assert len(set(got)) == 2 and slug in got
    assert any(g.startswith(slug + "-") for g in got)
