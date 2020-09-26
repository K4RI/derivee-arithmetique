def isprime(u):
    if u in (0,1):
        return False
    if u==2:
        return True
    for i in ((2,)+tuple(range(3,int(u**0.5)+1,2))):
        if u%i==0:
            return False
    return True



Nmax=10**4
nun=0

# P=[u for u in range(Nmax) if isprime(u)]
print('ok')

# print(moy_th(Nmax))

colors = ['g','b','c','m', 'y']
from random import choice

def decomp_primfacts(n):
    if n==0:
        return 'erreur'
    if n==1:
        return []
    for p in P:
        # print(p)
        alph=0
        while n%p==0:
            n=n/p
            alph+=1
            # print('aaa',alpha)
        if alph!=0:
            return [(p,alph)] + decomp_primfacts(n)


def derivee_arith(n):
    global nun
    nun=n
    if nun%(10**5)==0:
        print(nun)  
    if n==0:
        return 0
    decomp=decomp_primfacts(n)
    return round(n*sum([alph/p for (p,alph) in decomp]))


from fractions import Fraction

def pente(n):
    decomp=decomp_primfacts(n)
    return sum([Fraction(alph,p) for (p,alph) in decomp])

def pp(n):
    print(pente(n))

def derivee_arith_frac(n):
    return n*pente(n)


from matplotlib import pyplot as plt

X=[i for i in range(1,Nmax)]

def plot_d():
    "affiche le graphe de d"
    Y=[derivee_arith(x) for x in X]
    plt.plot(X,Y, 'r.', markersize=2)

def plot_d_seconde():
    "affiche le graphe de d²"
    Y=[derivee_arith(derivee_arith(x)) for x in X]
    Yjej1=[3/2*x +12 for x in X]
    Yjej2=[x/6 + 11/3 for x in X]
    Yjej3=[x/6 + 7/2 for x in X]
    plt.plot(X,Y, 'r.', markersize=3)
    plt.plot(X,Yjej1, 'g', linewidth=1)
    plt.plot(X,Yjej2, 'g', linewidth=1)
    plt.plot(X,Yjej3, 'g', linewidth=1)

def moy_th(N):
    sum=0
    for p in P:
        if p<=N:
            sum+=1/(p*(p-1))
    return sum

Y=[derivee_arith(x)/x for x in X]
# Y2 = [y for y in Y if y<=3]

import numpy as np
from matplotlib.ticker import PercentFormatter
def plot_d_norm():
    "affiche le graphe de d_norm"
    print('ok Y')
    
    Yt=[]
    MED=[]
    MOY=[]
    lim_moy=moy_th(Nmax)
    print('ok th')
    sum=0
    for n in range(0, Nmax-1):
        sum+=Y[n]
        MOY.append(sum/(n+1))
        
        Yt.append(Y[n])
        Yt.sort()
        if n%2==0:
            MED.append(Yt[n//2])
        else:
            MED.append((Yt[n//2]+Yt[n//2 + 1])/2)
    Y2=Y.copy()
    Y2.sort()
    print(Y2[len(Y2)//2])
    
    
    plt.figure(1)
    plt.plot(X,Y, 'r.', markersize=1)
    plt.plot(X,MOY, 'green', label='moyenne $M_N$', linewidth=2)
    # plt.plot(X,(Nmax-1)*[lim_moy], 'b-.', label='limite théorique de $M_N$', linewidth=2)
    plt.plot(X,MED, 'black', label='médiane $m_N$', linewidth=2)
    plt.xlim(-0.01*Nmax, 1.01*Nmax)
    plt.ylim(-0.05, 2) 
    
    
    # plt.figure(1)
    # plt.hist(Y2, 1500,  weights=np.ones(len(Y2)) / len(Y2), color = 'pink', edgecolor = 'green')
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # print('ok hist')
    # plt.xlim(-0.05, 3)




from math import log2, log

def encadre_d():
    "affiche les fcts qui encadrent le graphe de d"
    Ym=[2*x**0.5 for x in X] # diviser Ym et YM par x pour avoir les encadrements de d_norm
    plt.plot(X,Ym, 'blue', linewidth=1, label='points $p^2$ sur $x \mapsto 2 \sqrt{n}$')
    X1 = [p*p for p in P if p*p<Nmax]
    Y1=[derivee_arith(x) for x in X1]
    plt.plot(X1, Y1, 'b.', markersize=5)


    YM=[x*log2(x)/2 for x in X]
    plt.plot(X,YM, 'green', linewidth=1, label='points $2^n$ sur $x \mapsto n \log_2(n) / 2$')
    X2 = [2**n for n in range(int(log2(Nmax)))]
    Y2=[derivee_arith(x) for x in X2]
    plt.plot(X2, Y2, 'g.', markersize=5)




def courbes_B(l,m):
    "les courbes logarithmiques sur le graphe de d_norm passant par les n*P[i]^k"
    couleurrs=['b', 'g', 'k', 'y']
    
    for i in range(m):
        loglel=log(P[i])
        st=str(P[i])
        labl= 'ensembles des $n%d^k$ sur y=$log_{%d}(x/n) / %d + d_{norm}(n)$ pour 1 <= n <= %d' %(P[i], P[i], P[i], l)
        plt.plot([0,1],[1, 1.001], couleurrs[i], linewidth=0.5, label=labl)
        for n in range(1,l):
            YM=[log(x/n)/(P[i]*log(P[i])) + pente(n) for x in X]
            X1=[n*P[i]**k for k in range(1, int(log(Nmax/n)/loglel)+1 ) ]
            Y1=[derivee_arith(x)/x for x in X1]
            plt.plot(X,YM, couleurrs[i], linewidth=0.5)
            plt.plot(X1, Y1, couleurrs[i]+'.', markersize=7)


def plot_An(n, color='g'):
    "affiche l'affine {(np, p premier)}"
    X1 = [n*p for p in P if n*p<Nmax]
    Y1=[derivee_arith(x) for x in X1]
    labl1 = '$A_{' + str(n) + '},$' + '$y=' + str(pente(n)) + 'x+' + str(n) + '$'
    plt.plot(X1, Y1, color+'.', markersize=7)
    plt.plot([0,Nmax], [n, pente(n)*Nmax+n], color, linewidth=1, label=labl1)


def plot_An_inter(n, p, col):
    "pareil mais sert à l'affichage de plot_inters"
    r=n//p
    X1 = [r*p for p in P if r*p<Nmax]
    Y1=[derivee_arith(x) for x in X1]
    labl= '$A_{%d/%d}=A_{%d}, y=' %(n,p,r) + str(pente(r)) + 'x+%d$' % (r)
    col=choice(colors)
    plt.plot([0,Nmax], [r, pente(r)*Nmax+r], col, linewidth=1, label=labl)
    plt.plot(X1, Y1, col+'.', markersize=7)

def plot_inters(n):
    "affiche tous les Ak auquel n appartient"
    decomp=decomp_primfacts(n)
    P1=[p for (p, alph) in decomp]
    i=0
    for p in P1:
        col=colors[i]
        plot_An_inter(n, p, col)
        i+=1
    labl= '$(%d, d(%d))$' %(n, n)
    plt.plot(n, derivee_arith(n), 'k*', markersize=15, label=labl)


def plot_Aprn(n, color='g'):
    X1 = [n*p for p in P if n*p<Nmax]
    Y1=[derivee_arith(x)/x for x in X1]
    lim=derivee_arith(n)/n
    Y1t=[lim + n/x for x in X]
    labl1 = "$A'_{%d}, y=" %(n) + str(pente(n)) + " + %d/x$" %(n)
    plt.plot(X1, Y1, color+'.', markersize=7)
    plt.plot(X, Y1t, color, linewidth=1, label=labl1)


# plot_d()
# encadre_d()
# plot_An(8)
# plot_inters(11*57)


plot_d_norm()

# plot_Aprn(8)
# courbes_B(12, 2)

# plot_d_seconde()


plt.legend(loc='upper left')

# plt.xlim(-10, 10010)
# plt.ylim(-0.1,6.6)

plt.show()



# for i in range(1,100):
#     print(i, derivee_arith(i), pente(i))


##
def verif(n):
    print(n, decomp_primfacts(n))
    a=derivee_arith(n)
    print(a, decomp_primfacts(a))
    print(derivee_arith(a), derivee_arith(a)-n/6, '\n')

# les regroupements de (n, d(d(n)) sur des droites affines
print('\n 3/2X + 12 ->  8*(un p tq 3p+2 est premier)')
verif(2504)
verif(2536)
verif(8312)

print('\n X/6 + 7/2 -> 3* (2p-3)')
verif(6297) # + 7/2
verif(7149) # + 7/2

print('\n X/6 + 11/3 -> 2* (3p-2)')
verif(6914)
verif(7274)

##
N= 10**5


P=[u for u in range(N) if isprime(u)]
cpt=0
for p in P:
    if isprime(3*p+2):
        # print(p)
        cpt+=1
n=len(P)
print(N, ':', cpt, n)
print('Entre 1 et ' + str(N) + ', ' + str(100*cpt/n) + '% des premiers sont tels que 3p+2 est aussi premier')

# https://oeis.org/A088878

# pi_1(n)=card{les premiers congrus à 2 mod 3)

# -> pi_1(n)~pi(n)/2

##
limth=0
for p in P:
    limth+=1/(p*(p-1))
print('La limite théorique de la moyenne est ' + str(limth))




##

from matplotlib import pyplot as plt

def isprime(u):
    if u in (0,1):
        return False
    if u==2:
        return True
    for i in ((2,)+tuple(range(3,int(u**0.5)+1,2))):
        if u%i==0:
            return False
    return True


P2=[u for u in range(10**3) if isprime(u)]

X=[i for i in range(len(P2))]
Ytest=[i*P2[-1]/len(P2) for i in range(len(P2))]

plt.plot(X, P2, 'r.', markersize=1)
plt.plot(X, Ytest)
plt.show()

##
N= 10**5


P=[u for u in range(N) if isprime(u)]
cpt=0
for p in P:
    if isprime(3*p+2):
        # print(p)
        cpt+=1
n=len(P)
print(N, ':', cpt, n)
print('Entre 1 et ' + str(N) + ', ' + str(100*cpt/n) + '% des premiers sont tels que 3p+2 est aussi premier')

# https://oeis.org/A088878

# pi_1(n)=card{les premiers congrus à 2 mod 3)

# -> pi_1(n)~pi(n)/2

##
N = 3*10**5

P=[1]+[u for u in range(N) if isprime(u)]
print('ok')

def U(n,p):
    return int(log(n)/log(p))


def summ(N):
    sum=0
    for p in P:
        if p<=N:
            sum+=(U(N,p)*(p-1)+1)/(p**U(N,p)*(p-1))
    return sum

def summ2(N):
    sum=0
    for k in range(2,len(P)):
        p=P[k]
        if p<=N:
            sum+=1/(((k*log(k))**U(N,p))*k*k)
    return log(N)*sum


X=[i for i in range(2, N, 1000)]
Y=[summ2(x) for x in X]

plt.plot(X,Y)
plt.show()





