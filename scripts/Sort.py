# author = Alexandros Ioannidis

# Python code to sort the tuples using first element of sublist Inplace way to sort using sort()
def Sort(sub_li):
    sub_li.sort(key=lambda x: x[0])
    return sub_li
