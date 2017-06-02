============
Contribution
============

You can contribute to this repository your own code and improve the existing one.
Please read this section carefully to hold a certain standard in code quality.

Commit Messages
===============
The commit messages follow the
`AngularJS Git Commit Message Conventions <https://gist.github.com/stephenparish/9941e89d80e2bc58a153>`_
with the following format:

``<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>``

Usually the subject line (``<type>: <subject>``) is enough.
It contains a succinct description of the change. Allowed ``<type>`` are:

 * feat (feature)
 * fix (bug fix)
 * docs (documentation)
 * style (formatting, missing semi colons, …)
 * refactor
 * test (when adding missing tests)
 * chore (maintain)

An example would be: ``feat: Dice coefficient``

Code Documentation
==================
Please document your code. Each package, module, class, and function should have a comment.
Don't forget to add a documentation to Sphinx (reStructuredText).

Code Style
==========
We follow the `PEP 8 -- Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_.

Tests
=====
You do write tests, don't you? =) They are located in the ``./test`` directory
