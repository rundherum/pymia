# Medical Image Analysis Python Common Repository
This repository contains Python modules, classes, and functions useful for medical image analysis.

 * [Installation](#installation)
 * [Repository Content](#repository-content)
 * [Contribution](#contribution)

## Installation
The library can be built using CMake...

Add the Boost and ITK paths to your environment:
```
export ITK_DIR="/usr/local/..."
export BOOST_ROOT="/usr/local/boost/boost_1_64_0"
```

### Dependencies
The source code depends on two external libraries:

 - [Boost C++ Libraries](www.boost.org/)
 - [Simple Insight Segmentation and Registration Toolkit (sITK)](https://itk.org/)

## Repository Content


## Contribution
You can contribute to this repository your own code and improve the existing one. Please read this section carefully to hold a certain standard in code quality.

### Commit Messages
The commit messages follow the [AngularJS Git Commit Message Conventions](https://gist.github.com/stephenparish/9941e89d80e2bc58a153) with the following format:

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```
Usually the subject line (```<type>: <subject>```) is enough. It contains a succinct description of the change. Allowed ```<type>``` are:
 - feat (feature)
 - fix (bug fix)
 - docs (documentation)
 - style (formatting, missing semi colons, …)
 - refactor
 - test (when adding missing tests)
 - chore (maintain)

An example would be: ```feature: Dice coefficient```

### Code Documentation
Please document your code. Each class and function should have a comment.

### Code Style
We mostly follow the PEP and Google code guideline for Python.

### Tests
You do write tests, don't you? =)
