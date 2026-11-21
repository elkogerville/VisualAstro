Submitting a Pull Request
=========================

Workflow
--------

1. Sync with upstream:

   .. code-block:: bash

      git fetch upstream
      git switch main
      git merge upstream/main

2. Create a branch:

   .. code-block:: bash

      git switch -c <branch-name>

3. Commit, then push to your fork:

   .. code-block:: bash

      git push -u origin <branch-name>

4. Open a pull request against ``elkogerville/VisualAstro``.


Pre-Merge Checklist
--------------------

All tests must pass, and the docs must build! They are automatically
run by a github workflow when you create a PR and each time you commit.
A VisualAstro maintainer will then need to approve your PR before you can merge it.
After this you will be an official VisualAstro contributor! Thank you for supporting our package!
