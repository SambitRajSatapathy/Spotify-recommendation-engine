#!/usr/bin/env python
# coding: utf-8

# In[8]:


list1 = []
for number in range(1, 15):
    items = int(input())
    list1.append(number)


# In[9]:


list1


# In[15]:


n = int(input("Number of elements  "))
# arr = map(int, input().split())
list1 = []
for number in range(1, n+1):
    elements = int(input("elements : "))
    list1.append(elements)


# In[16]:


list1


# In[18]:


if __name__ == '__main__':
    alist = []
    for i in range(int(input())):
        name = input()
        score = float(input())
        alist.append([name, score])
second_highest = sorted(set([score for name, score in alist]))
print('\n'.join(sorted([name for name, score in alist if score == second_highest])))


# In[21]:


str(hash(10))


# In[20]:


hash(150
    )


# In[23]:


n = int(input())
Tuple1 = map(int, input().split())
Tuple1


# In[40]:


n = int(input())
l1 = []
for i in range(n):
    j = int(input())
    l1.append(j)
    t = tuple(l1)
print(hash(t))


# In[33]:


l1


# In[41]:


n = int(input())
Tuple1 = map(int, input().split())
t = tuple(Tuple1)
print(hash(t))


# In[44]:


n = int(input())
integer_list = map(int, input().split())
t = tuple(integer_list)
result = hash(t)
print(result)


# In[53]:


n = int(input("numbers : "))
s = set(map(int, input().split()))
for i in s:
    command = input().split()
    if command[0] == "discard":
        s.discard(command[1])
    elif command[0] == "remove":
        s.remove(command[1])
    else:
        s.pop()


# In[50]:


s3


# In[ ]:


if __name__ == '__main__':

    N = int(input())

    List=[];

    for i in range(N):

        command=input().split();

        if command[0] == "insert":

            List.insert(int(command[1]),int(command[2]))

        elif command[0] == "append":

            List.append(int(command[1]))

        elif command[0] == "pop":

            List.pop();

        elif command[0] == "print":

            print(List)

        elif command[0] == "remove":

            List.remove(int(command[1]))

        elif command[0] == "sort":

            List.sort();

        else:

            List.reverse();


# In[56]:


M = int(input())
A = set(map(int, input().split()))
N = int(input())
B = set(map(int, input().split()))

l = sorted((A.difference(B)).union(B.difference(A)))

print ("\n".join([str(element) for element in l]))


# In[ ]:




