"""Credentials resolve as a pair: a configured token never travels to another host."""

from lb_anythings.application.project_context import NO_CREDENTIALS, Credentials

CONFIGURED = Credentials(hostname="http://ls-configured:8080/", access_token="cfg")


def test_nothing_from_setup_falls_back_to_the_configured_pair() -> None:
    assert NO_CREDENTIALS.resolved_against(CONFIGURED) == Credentials(
        "http://ls-configured:8080/", "cfg"
    )


def test_a_complete_pair_from_setup_is_used_as_is() -> None:
    setup = Credentials(hostname="http://ls-setup:8080", access_token="setup")

    assert setup.resolved_against(CONFIGURED) == setup


def test_a_hostname_setup_introduced_never_borrows_the_configured_token() -> None:
    setup = Credentials(hostname="http://ls-setup:8080")

    assert setup.resolved_against(CONFIGURED) == Credentials("http://ls-setup:8080", None)


def test_the_configured_token_serves_the_configured_host_however_it_is_spelled() -> None:
    setup = Credentials(hostname="http://LS-CONFIGURED:8080")

    assert setup.resolved_against(CONFIGURED).access_token == "cfg"


def test_a_token_without_a_hostname_is_used_against_the_configured_host() -> None:
    setup = Credentials(access_token="setup")

    assert setup.resolved_against(CONFIGURED) == Credentials("http://ls-configured:8080/", "setup")


def test_resolving_against_nothing_configured_changes_nothing() -> None:
    setup = Credentials(hostname="http://ls-setup:8080", access_token="setup")

    assert setup.resolved_against(NO_CREDENTIALS) == setup
