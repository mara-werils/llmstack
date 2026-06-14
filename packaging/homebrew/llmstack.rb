class Llmstack < Formula
  include Language::Python::Virtualenv

  desc "Local-first AI coding tool — chat with any codebase, privately, for free"
  homepage "https://github.com/mara-werils/llmstack"
  # url + sha256 are kept in sync automatically by the release workflow
  # (.github/workflows/release.yml → update-homebrew-tap). The sha256 below is a
  # placeholder for the checked-in copy; the published tap always carries the
  # real digest for the tagged release.
  url "https://github.com/mara-werils/llmstack/archive/refs/tags/v1.0.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "Apache-2.0"

  depends_on "python@3.12"

  def install
    # Install the CLI and its runtime dependencies into an isolated virtualenv,
    # then link the `llmstack` entry point onto PATH.
    venv = virtualenv_create(libexec, "python3.12")
    venv.pip_install_and_link buildpath
  end

  test do
    assert_match "llmstack", shell_output("#{bin}/llmstack --version")
  end
end
