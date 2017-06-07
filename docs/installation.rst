============
Installation
============



Dependencies
------------
dsf


ITK
---

You can install ITK using pip:

.. code-block:: bash

    pip install itk -f https://github.com/InsightSoftwareConsortium/ITKPythonPackage/releases/tag/latest


Or... the longer and more painful way...
Some modules depend on ITK. This describes how to install ITK with wrappers for Python (tested with ITK 4.12).

You will need ccmake (install by sudo apt-get install cmake-curses-gui)

1. Download ITK https://itk.org/ITK/resources/software.html
2. Create ITK install folder

.. code-block:: bash

    cd /usr/local
    sudo mkdir itk

3. Extract ITK

.. code-block:: bash

    sudo tar xvzf InsightToolkit-4.9.1.tar.gz -C /usr/local/itk
    sudo mkdir /usr/local/itk/Insight.../bin

4. Start the configuration

.. code-block:: bash

    cd /usr/local/itk/Insight.../bin
    sudo ccmake ..

Press the *c* key to configure the build. We need to change now some options
(press *t* to active the advanced mode if you don't see the options).::

    BUILD_SHARED_LIBS       ON
    Module_BridgeNumPy      ON
    ITK_WRAP_PYTHON         ON
    ITK_LEGACY_SILENT       ON
    BUILD_EXAMPLES          OFF
    BUILD_TESTING           OFF

Configure now again by pressing *c* and ignore the warning after the configuration by pressing *e*.
Now set the following wrapping options::

    ITK_WRAP_DIMS           2;3;4
    ITK_WRAP_float          ON
    ITK_WRAP_double         ON
    ITK_WRAP_signed_char    ON
    ITK_WRAP_signed_long    ON
    ITK_WRAP_signed_short   ON
    ITK_WRAP_unsigned_char  ON
    ITK_WRAP_unsigned_long  ON
    ITK_WRAP_unsigned_short ON
    WRAP_<data-type>        Select yourself which more to activate.

In case you want to use ITK with anaconda, you need to change the Python options::

    PYTHON_EXECUTABLE       /usr/local/anaconda/anaconda3/envs/<yourenv>/bin/python3.6
    PYTHON_INCLUDE_DIR      /usr/local/anaconda/anaconda3/envs/<yourenv>/include/python3.6m
    PYTHON_LIBRARY          /usr/local/anaconda/anaconda3/envs/<yourenv>/lib/libpython3.so
    PY_SITE_PACKAGES_PATH   /usr/local/anaconda/anaconda3/envs/<yourenv>/lib/python3.6/site-packagesc

If you install ITK for the entire system, verify the Python options::

    PYTHON_EXECUTABLE       /usr/bin/python
    PYTHON_INCLUDE_DIR      /usr/include/x86_64-linux-gnu/python2.7
    PYTHON_LIBRARY          /usr/lib/x86_64-linux-gnu/libpython2.7.so.1.0
    PY_SITE_PACKAGES_PATH   /usr/lib/python2.7/dist-packages

To finish the configuration press *c* again and then generate the make file by pressing *g*.

5. Compile and install

.. code-block:: bash

    sudo make -j<number-of-your-processors>
    sudo make install


https://itk.org/ITKSoftwareGuide/html/Book1/ITKSoftwareGuide-Book1ch3.html#x32-420003.7