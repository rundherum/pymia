Use with PyCharm
================

We recommend JetBrain's PyCharm as IDE for Python projects. It is free to use, so give it a try.
PyCharm can be downloaded without cost under ...
You can easily

Before you use PyCharm, ensure that you have been added

1. Open PyCharm
2. In the main menu select VCS > Checkout from Version Control > GitHub
3. The Clone Repository dialog appears. Select the https://github.com/istb-mia/miapy.git
and specify the parent directory and directory name (miapy)

Use miapy in another project
-----------------------------

If you have a existing project and you would like to use miapy, you can add it as additional project.

1. Open your project with PyCharm
2. On the main menu choose *File > Open...*
3. In the Open File or Project dialog, select the miapy project directory and click OK, the Open Project dialog appears
4. Select the option open in current window and tick the Add to currently opened projects

You can now use the packages and modules, e.g.:

from miapy.evaluation.evaluator import Evaluator, ConsoleEvaluatorWriter

Moreover, you can directly contribute. If you add, change, or remove files and commit them with VCS...

Refer also to the PyCharm help https://www.jetbrains.com/help/pycharm/opening-multiple-projects.html