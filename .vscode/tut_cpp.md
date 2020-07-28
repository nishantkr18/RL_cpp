## Compilation Links
- Everything you want to know about GCC (https://medium.com/@meghamohan/everything-you-want-to-know-about-gcc-fa5805452f96)
- Everything you need to know about Libraries in C (https://medium.com/@meghamohan/everything-you-need-to-know-about-libraries-in-c-e8ad6138cbb4)

## To keep in mind:

- including -L along with -l (https://www.cs.swarthmore.edu/~newhall/unixhelp/howto_C_libraries.html) 


## Uninstalling mlpack from system:
```
# remove lib
rm  /usr/local/lib/libmlpack*

# remove header files
rm -r /usr/local/include/mlpack

# remove executables
rm /usr/local/bin/mlpack_*
```
## What is the purpose of the -g flag when compiling the C/C++ program? 
- It tells the compiler to emit debug information. This information is used to help the debugger know details which can normally only be known from the source code