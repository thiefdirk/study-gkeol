class Sum :
    a1 = 0
    a2 = 0
    def number(self, x, y) : 
     self.a1 = x
     self.a2 = y
    def sum(self) :
     print(self.a1)
     print('+')
     print(self.a2)
     print('=')
     print(self.a1 + self.a2)
    
a = Sum()

a.number(1,2)
a.sum()


a.number(2,1937)
a.sum()

def sum(c,d) :
    print(c)
    print('+')
    print(d)
    print('=')
    return c+d

print(sum(5, 4894))
 