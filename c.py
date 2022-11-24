x=2
print(x)
#x=~2
#print(x)
print(2^(~2))
a = ~5
#this will print a in binary
bnr = bin(a).replace('0b','')
x = bnr[::-1] #this reverses an array
while len(x) < 8:
    x += '0'
bnr = x[::-1]
print(bnr)
a = 5
#this will print a in binary
bnr = bin(a).replace('0b','')
x = bnr[::-1] #this reverses an array
while len(x) < 8:
    x += '0'
bnr = x[::-1]
print(bnr)
print(bin(~11).replace('0b',''))