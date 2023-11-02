import math
from gpoly import Poly
import helper
import random
import datetime
import time
import math
import sys

# https://eprint.iacr.org/2012/688.pdf
class Ding12(object):
    def __init__(self, n, q, sigma):
        self.n = n
        self.q = q
        self.sigma = sigma
        self.a = Poly.uniform(self.n, self.q)
        self.sA = Ding12.gen_secret(self.n, self.q, self.sigma)
        self.eA = Ding12.gen_pubkey_error(self.n, self.q, self.sigma)
        self.sB = Ding12.gen_secret(self.n, self.q, self.sigma)
        
    @staticmethod
    def calculate_sigma(n, q, alpha):
        return alpha * q / math.sqrt(n)

    @staticmethod
    def gen_secret(n, q, sigma):
        return Poly.discretegaussian(n, q, sigma)

    @staticmethod
    def gen_pubkey_error(n, q, sigma):
        return Poly.discretegaussian(n, q, sigma)

    @staticmethod
    def gen_shared_error(n, q, sigma,pA=0):
        if pA:
            seed = hash(str(pA))
        return Poly.discretegaussian(n, q, sigma,seed)

    @staticmethod
    def sig(q, v):
        # v = v % q
        if v > round(q / 4.0) and v < q - math.floor(q / 4):
            return 1
        else:
            return 0

    def signal(self, v):
        return [Ding12.sig(self.q, v[i]) for i in range(self.n)]

    def mod2(self, v, w):
        r = [0 for i in range(self.n)]
        for i in range(self.n):
            r[i] = int(v[i] + w[i] * (self.q - 1) / 2)
            r[i] %= self.q
            r[i] %= 2
        return r

    def alice_init(self):
        self.pA = self.a * self.sA + 2 * self.eA
        return self.pA

    def bob(self, pA):
        self.eB = Ding12.gen_pubkey_error(self.n, self.q, self.sigma)
        self.gB = Ding12.gen_shared_error(self.n, self.q, self.sigma, pA)
        self.pB = self.a * self.sB + self.eB + self.eB
        self.kB = pA * self.sB + self.gB + self.gB
        self.wB = self.signal(self.kB)
        self.skB = self.mod2(self.kB, self.wB)
        return (self.pB, self.wB, self.skB)
    
    def alice_resp(self, pB, wB): # runtime : n^2
        self.gA = Ding12.gen_shared_error(self.n, self.q, self.sigma)
        self.kA = pB * self.sA + self.gA + self.gA
        self.skA = self.mod2(self.kA, wB)
        return self.skA


def getabs_sBp(n, q, sB, signals, l, param):

    if l == 4: code = [0, 8, 4, 12, 2, 10, 6, 14, 3, 11, 7, 15, 1, 9, 5, 13]
    if l == 5: code = [0, 16, 8, 24, 4, 20, 12, 28, 6, 22, 14, 30, 2, 18, 10, 26, 3, 19, 11, 27, 7, 23, 15, 31, 5, 21, 13, 29, 1, 17, 9, 25]
    if l == 6: code = [0, 48, 24, 40, 12, 60, 20, 36, 6, 54, 30, 46, 10, 58, 18, 34, 3, 51, 27, 43, 15, 63, 23, 39, 5, 53, 29, 45, 9, 57, 17, 33, 1, 49, 25, 41, 13, 61, 21, 37, 7, 55, 31, 47, 11, 59, 19, 35, 2, 50, 26, 42, 14, 62, 22, 38, 4, 52, 28, 44, 8]
    if l == 7: code = [0, 96, 48, 80, 24, 120, 40, 72, 12, 108, 60, 92, 20, 116, 36, 68, 6, 102, 54, 86, 30, 126, 46, 78, 10, 106, 58, 90, 18, 114, 34, 66, 3, 99, 51, 83, 27, 123, 43, 75, 15, 111, 63, 95, 23, 119, 39, 71, 5, 101, 53, 85, 29, 125, 45, 77, 9, 105, 57, 89, 17, 113, 33, 65, 1, 97, 49, 81, 25, 121, 41, 73, 13, 109, 61, 93, 21, 117, 37, 69, 7, 103, 55, 87, 31, 127, 47, 79, 11, 107, 59, 91, 19, 115, 35, 67, 2, 98, 50, 82, 26, 122, 42, 74, 14, 110, 62, 94, 22, 118, 38, 70, 4, 100, 52, 84, 28, 124, 44, 76, 8]
    
    
    # if l == 6: code = [0, 32, 24, 56, 8, 44, 20, 52, 6, 38, 30, 62, 14, 42, 18, 50, 2, 34, 26, 58, 10, 46, 22, 54, 4, 36, 28, 60, 12, 40, 16, 48, 1, 33, 25, 57, 9, 45, 21, 53, 7, 39, 31, 63, 15, 43, 19, 51, 3, 35, 27, 59, 11, 47, 23, 55, 5, 37, 29, 61, 13]
    # if l == 7: code = [0, 64, 48, 112, 24, 88, 40, 104, 12, 76, 60, 124, 20, 84, 36, 100, 6, 70, 54, 118, 30, 94, 46, 110, 10, 74, 58, 122, 18, 82, 34, 98, 3, 67, 51, 115, 27, 91, 43, 107, 15, 79, 63, 127, 23, 87, 39, 103, 5, 69, 53, 117, 29, 93, 45, 109, 9, 73, 57, 121, 17, 81, 33, 97, 1, 65, 49, 113, 25, 89, 41, 105, 13, 77, 61, 125, 21, 85, 37, 101, 7, 71, 55, 119, 31, 95, 47, 111, 11, 75, 59, 123, 19, 83, 35, 99, 2, 66, 50, 114, 26, 90, 42, 106, 14, 78, 62, 126, 22, 86, 38, 102, 4, 68, 52, 116, 28, 92, 44, 108, 8] 

    checkcode = []
    for i in range(n):
        sum = 0
        for b in range(l):
            sum += 2**(b) * signals[b][i]
        checkcode.append(sum)

    sBp = []
    for i in range(n):
        sBp.append(code.index(checkcode[i]))
    
    return sBp


def collect_signals(n, q, alpha, a, sB, istar, l):
    global execution
    global query

    pA = Poly(n, q)
    f  = Poly(n, q)
    f[0] = 1
    param = []
    signals = []
    if l == 5:
        f[istar] = 1
    k = [550, 1050, 4000, 8192]
    if l == 5: k = [260,525,1050,4000,8192]
    for i in range(l):
        pA[0] = k[i]
        param.append(pA[0])
        (pB, wB, skB) = execution.bob(pA * f)
        query += 1
        signals.append(wB)
    sBp = getabs_sBp(n, q, sB, signals, l, param)

    return sBp


def collect_abs(n, q, alpha, a, sB, istar, l):
    coeffs_abs = collect_signals(n, q, alpha, a, sB, istar, l)
    return coeffs_abs
    

def strarray(a):
    s = "["
    for x in a:
        if isinstance(x, bool): s += " T" if x else " F"
        elif isinstance(x, int): s += "{:2d}".format(x)
        elif isinstance(x, str): s += "{:2s}".format(x)
        elif x is None: s += "  "
        else: assert False, "Invalid type " + x
        s += " "
    s += "]"
    return s


def get_zeros(coeffs, n):
    zero_count = 0
    temp_count = 0
    for i in range(n):
        if coeffs[i] == 0:
            temp_count += 1
        else:
            if temp_count > zero_count:
                zero_count = temp_count
            temp_count = 0
    return zero_count


def fluhrer_attack_new(n, q, alpha, a, sB):
    global zero
    coeffs_abs = collect_abs(n, q, alpha, a, sB, 0, 4)

    zero_count = get_zeros(coeffs_abs,n)
    zero  += zero_count
    # print(zero_count)
    # zero_count = 4
    MAX_ISTAR = zero_count + 2
    coeffs = list(range(MAX_ISTAR))
    coeffs[0] = coeffs_abs
    for istar in range(1,MAX_ISTAR):
        coeffs[istar] = collect_abs(n, q, alpha, a, sB, istar, 5)
        # print("istar = ", istar, "coeffs[istar] = ", strarray(coeffs[istar]))
    sign_comp = [[None for j in range(n)] for istar in range(n)]
    for istar in range(1, MAX_ISTAR):
        for i in range(istar, n):
            if coeffs[0][i] != 0 and coeffs[0][i - istar] != 0:
                if coeffs[istar][i] == coeffs[0][i] + coeffs[0][i - istar]: sign_comp[i][i - istar] = "S"
                else: sign_comp[i][i - istar] = "D"
        for i in range(istar):
            if coeffs[0][i] != 0 and coeffs[0][(i - istar) % n] != 0:
                if coeffs[istar][i] == coeffs[0][i] - coeffs[0][(i - istar) % n]:
                    sign_comp[i][(i - istar) % n] = "S"
                else:
                    sign_comp[i][(i - istar) % n] = "D"
    coeffs_signed = Poly(n, q)
    for i in range(n): coeffs_signed[i] = coeffs[0][i]
    for i in range(0, n):
        if coeffs_signed[i] != 0:
            pos_votes = 0
            neg_votes = 0
            for j in range(0, i):
                if coeffs_signed[j] < 0:
                    if sign_comp[i][j] == "S": neg_votes += 1
                    if sign_comp[i][j] == "D": pos_votes += 1
                elif coeffs_signed[j] > 0:
                    if sign_comp[i][j] == "S": pos_votes += 1
                    if sign_comp[i][j] == "D": neg_votes += 1
            if neg_votes > pos_votes: coeffs_signed[i] *= -1
    # print("sB            = ", strarray(sB))
    # print("coeffs_signed = ", strarray(coeffs_signed))
    return sB == coeffs_signed or sB == -1 * coeffs_signed


def collect_new_abs(n, q, alpha, a, sB, Lambda): # new add
    global execution
    global query

    pA = Poly(n, q)
    f  = Poly(n, q)
    f[0] = 1
    
    for i in range(1, Lambda):
        f[1024-i] = -1
    param = []
    signals = []
    
    if Lambda <= 8 and Lambda >= 5:
        k = [129, 258, 516, 1032, 2065, 4130, 8192]
    if Lambda <= 4:
        k = [258, 516, 1033, 2067, 4164, 8192]
        
    for i in range(len(k)):
        pA[0] = k[i]
        param.append(pA[0])
        (pB, wB, skB) = execution.bob(pA * f)
        query += 1
        signals.append(wB)
        
     
    if Lambda <= 8 and Lambda >= 5:
    	sBp = getabs_sBp(n, q, sB, signals, 7, param)
    if Lambda <= 4:
    	sBp = getabs_sBp(n, q, sB, signals, 6, param)
    return (sBp, pB)


def Check(Sign, sB, AbsB, pB, a): # new add
    coeffs_signed = Poly(n, q)
    coeffs_signed1 = Poly(n, q)
    for i in range(len(AbsB)):
        coeffs_signed[i] = Sign[i]*AbsB[i]
        coeffs_signed1[i] = Sign[i]*AbsB[i]
    # print("Coeffs = ", coeffs_signed)
    # print("pB = ", pB)
    # return sB == coeffs_signed or sB == -1*coeffs_signed:

    summ = 0
    flag = 0
    coeffs_signed *= a
    for i in range(len(AbsB)):
        if abs((pB[i] - coeffs_signed[i])) < 31 or abs(pB[i] - coeffs_signed[i]) > 16354:
            summ += 1
    if summ == len(AbsB):
        print("sB = ", sB)
        print("Sign = ", Sign)
        flag = 1
    if flag == 1:
        if coeffs_signed1 != sB:
            print("No!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            exit()
            return 0
        return 1
    return 0


def Rec(AbsB, AbsEight, Sign, errors, num, Lambda):  #

    i = num
    if num != 0:
        i = num - (Lambda-2)
        Sign[(num)] *= -1
    while i < len(AbsB) - (Lambda-1):
        flag1 = 0
        flag2 = 0
        summ = 0
        for j in range(Lambda-1):
            summ += Sign[(i+j)]*AbsB[(i+j)]

        if abs(summ + AbsB[(i+Lambda-1)]) == AbsEight[i]:
            flag1 = 1
        if abs(summ - AbsB[(i+Lambda-1)]) == AbsEight[i]:
            flag2 = 1
            
        # Case 1
        if flag1 == 1 and flag2 == 0:
            Sign[(i+Lambda-1)] = 1
            
        # Case 2
        if flag2 == 1 and flag1 == 0:
            Sign[(i+Lambda-1)] = -1

        # Case 3
        if flag1 == 0 and flag2 == 0:
            if len(errors) > 0 and num == 0:
                temp = errors.pop()
                Sign[temp] *= -1
                i = temp - (Lambda-1)
            else:
                return 0

        # Case 4
        if flag1 == 1 and flag2 == 1:
            if AbsB[(i+Lambda-1)] == 0:
                Sign[(i+Lambda-1)] = 0
            else :
                # Randomly selects the sign of Sign[(i+Lambda-1)], and store the position.
                Sign[(i+Lambda-1)] = -1
                errors.append((i+Lambda-1))

        i += 1
    while i >= len(AbsB) - (Lambda-1) and i < len(AbsB):
        summ3 = 0
        for j in range(Lambda):
            if i + j < len(AbsB):
                summ3 += Sign[(i+j)%len(AbsB)]*AbsB[(i+j)%len(AbsB)]
            else :
                summ3 -= Sign[(i+j)%len(AbsB)]*AbsB[(i+j)%len(AbsB)]
        
        if abs(summ3) != AbsEight[i]:
            return 0
        i += 1
    return 1


def New_attack(n, q, alpha, a, sB, Lambda):
    global zero
    global query
    global candi_num
    global Max_candi_num
    global lenErrors
    ans_num = 0

    AbsB = collect_abs(n, q, alpha, a, sB, 0, 4)
    zero_count = get_zeros(AbsB,n)
    zero  += zero_count
    if zero_count == 1:
        return fluhrer_attack_new(n, q, alpha, a, sB)
    else:
        (AbsEight, pB)= collect_new_abs(n, q, alpha, a, sB, Lambda)
        
        zero_loc = []
    
        for i in range(Lambda-1):
            if AbsB[i] == 0:
                zero_loc.append(i)
    
        for i in range(pow(2, Lambda-1-len(zero_loc))):
            Sign = [0 for i in range(1024)]
            temp = i
            tim = Lambda-1 - len(zero_loc)
            loczero = 0
            while tim > 0:
                cnt = temp & 1
                temp >>= 1
                while loczero in zero_loc:
                    loczero += 1
                if cnt == 0:
                    Sign[loczero] = -1
                else:
                    Sign[loczero] = 1

                tim -= 1
                loczero += 1
            # print("Test = ", Sign[0:6])
            errors = []

            cnt = Rec(AbsB, AbsEight, Sign, errors, 0, Lambda)
            
            lenErrors += len(errors)

            if cnt == 1:  
                ReC = Check(Sign, sB, AbsB, pB, a)
                if ReC == 1:
                    # ans_num += 1
                    return 1
                while len(errors) > 0:
                    loc = errors.pop()
                    cnt = Rec(AbsB,  AbsEight, Sign, errors, loc, Lambda)
                    if cnt == 1:   
                        ReC1 = Check(Sign, sB, AbsB, pB, a)
                        if ReC1 == 1:
                            #ans_num += 1
                            return 1
            else:
                while len(errors) > 0:
                    loc = errors.pop()
                    cnt = Rec(AbsB,  AbsEight, Sign, errors, loc, Lambda)
                    if cnt == 1:   
                        ReC1 = Check(Sign, sB, AbsB, pB, a)
                        if ReC1 == 1:
                            #ans_num += 1
                            return 1
        #print("Num = ", ans_num)
        return 0
        #return ans_num == 1


if __name__ == "__main__":
    n = 1024
    q = 16385
    sigma = 3.197
    t = 100
    alltime = 0
    succ = 0
    count = 1000000
    
    Lambda = int(sys.argv[1])
    
    if Lambda > 8 or Lambda < 3:
        print("Lambda must be selected from {3,4,5,6,7,8}")
        exit()
    
    print("parameters: n = {:d}, q = {:d}, sigma = {:f}, lambda = {:d}".format(n, q, sigma, Lambda))

    global query
    global zero
    global lenErrors
    
    lenErrors = 0
    query = 0
    zero = 0
    now = int(time.time())
    for seed in range(count):
        start = time.time()
        print("======================")
        
        print("seed = ", seed)
        if seed != 0:
            print("succ:", succ / seed)

        random.seed(seed + now) 

        seed += now
        a = Poly.uniform(n, q)
        sB = Poly.discretegaussian(n, q, sigma, seed)

        global execution
        execution = Ding12(n, q, sigma)
        execution.a = a
        execution.sB = sB

        if New_attack(n, q, sigma, a, sB, Lambda) == True:
            succ += 1
            
        end = time.time()
        cost = end - start
        alltime += cost

    print("succ:", succ / count)
    print("all queries: ", query)
    print("average: {} {} {} {}".format(alltime / count, query / count, zero / count, lenErrors / count))
    print("Lmabda = ", Lambda)
