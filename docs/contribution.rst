============
Contribution
============

You can contribute to this repository your own code and improve the existing one.
Please read this chapter carefully to hold a certain standard in code quality.

Code Style
----------
We follow the `PEP 8 -- Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_.

Code Documentation
------------------
Please document your code. Each package, module, class, and function should have a comment.
For major changes it might also be good to update the documentation you are currently reading.
It is generated with `Sphinx <http://www.sphinx-doc.org>`_ and you can find the source files in the ``./docs`` directory.

Code Tests
----------
You do write tests, don't you? They are located in the ``./test`` directory.

Commit Messages
---------------
The commit messages follow the
`AngularJS Git Commit Message Conventions <https://gist.github.com/stephenparish/9941e89d80e2bc58a153>`_
with the following format::

    <type>(<scope>): <subject>
    <BLANK LINE>
    <body>
    <BLANK LINE>
    <footer>

Usually the first line is enough, i.e. ``<type>: <subject>`` .
It contains a succinct description of the change. Allowed ``<type>`` s are:

 * feat (feature)
 * fix (bug fix)
 * docs (documentation)
 * style (formatting, missing semi colons, …)
 * refactor
 * test (when adding missing tests)
 * chore (maintain)

An example would be: ``feat: Dice coefficient``
