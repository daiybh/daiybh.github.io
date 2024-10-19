---
layout: post
title: Creating Pre-built Binary Packages
categories: Conan
keywords: conan, document
tags: [conan, document]
description: Creating Pre-built Binary Packages
---

this page is show "Creating Pre-built Binary Packages "

# Objective

To package pre-built binaries (such as .lib, .h) into a Conan package and publish it to the Conan Center, so that other projects can use these libraries directly.


<!--more-->
# Reference:

* [testPorject](https://github.com/daiybh/conan_test_prebuilt_binaries)

* [Conan Official Documentation: Creating Pre-built Binary Packages](https://docs.conan.io/2/tutorial/creating_packages/other_types_of_packages/package_prebuilt_binaries.html)

Assume we have a third-party library **myhellolib** that we want to use in our project.


# Steps

1. create a  local Library  **myhellolib**
2. create a pre-build Binary Package  **Prebuild_binaries**
3. create a Test Application **testApp** , this app use "Prebuild_binaries", check if it can work


# Detailed Steps

## 1. create a  local Library  **myhellolib**

```sh
mkdir myhellolib
cd myhellolib
conan new cmake_lib -d name=myhellolib -d version=1.0
conan build . -s build_type=Debug
```
at this point , you will have the following files

- build/libmyhellolib.a
- include/myhellolib.h

## 2. create a Pre-built Binary Package **Prebuild_binaries**

Use the Conan official example  "[Prebuild_binaries](https://github.com/conan-io/examples2/tree/main/tutorial/creating_packages/other_packages/prebuilt_binaries)" as a template

### Modify the  "conanfile.py" file:

```python
name = "myhellolib"
version = "0.3"

self.cpp_info.libs = ["myhellolib"]
```

### Copy the library files and header files

**NOTICE  my test os is macos armv8, you should change it according to your os**

copy myhellolib/build/libmyhellolib.a to Prebuild_binaries/vendor_hello_library/macos/armv8/libmyhellolib.a
copy myhellolib/include/myhellolib.h to Prebuild_binaries/vendor_hello_library/macos/armv8/include

### Publish to  the  Conan Center

```sh    
cd Prebuild_binaries 
```

chose a correct command match your os  and arch

```sh
conan export-pkg . -s os="Linux" -s arch="x86_64"
conan export-pkg . -s os="Linux" -s arch="armv8"
conan export-pkg . -s os="Macos" -s arch="x86_64"
conan export-pkg . -s os="Macos" -s arch="armv8"
conan export-pkg . -s os="Windows" -s arch="x86_64"
conan export-pkg . -s os="Windows" -s arch="armv8"
```
    
### Verify the publication:

```sh
conan list "myhellolib"
conan list "myhellolib/0.3#:*"
```

## 3. Create a Test Application **testApp**

```sh
mkdir testApp
cd testApp
conan new cmake_exe -d name=testApp -d version=1.0
```

### add "myhellolib"

1. Modify the conanfile.py file to add the dependency on myhellolib:

``` python

def requirements(self):
    self.requires("myhellolib/0.3")
```        
2. Modify the CMakeLists.txt file to link myhellolib:

```cmake
find_package(myhellolib)
target_link_libraries(testApp PRIVATE myhellolib::myhellolib)
```
     
3. Modify the src/main.cpp file to include the myhellolib header and call its function:

```cpp
#include "myhellolib.h"

int main(){ 
    ....
    myhellolib();
    ....
}
```
    
### Build the test application

    conan build . --output-folder=build --build=missing  -s build_type=Debug
     
     
### Run the test application

     ./bin/testApp
     
your will see the output from **myhellolib**