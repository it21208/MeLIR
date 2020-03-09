# author = Alexandros Ioannidis

# Python code to sort the tuples using first element of sublist Inplace way to sort using sort().


def Sort(sub_li):
    sub_li.sort(key=lambda x: x[0])
    return sub_li


# Python function to sort a list of sublists based on the 2nd element of all sublists.
def Sort_list_using_2nd_element_of_sublists(sub_list):
    l = len(sub_list)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_list[j][1] < sub_list[j + 1][1]):
                tempo = sub_list[j]
                sub_list[j] = sub_list[j + 1]
                sub_list[j + 1] = tempo
    return(sub_list)


# def SortedList(sub_li):
#  l = len(sub_li)
#  for i in range(0, l):
#    for j in range(0, l-i-1):
#      if (sub_li[j][1] > sub_li[j + 1][1]):
#        tempo = sub_li[j]
#        sub_li[j] = sub_li[j + 1]
#        sub_li[j + 1] = tempo
#  return(sub_li)
