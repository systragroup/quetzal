
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Name:        CalcDistribution
# Purpose:     Utilities for various calculations of different types of trip distribution models.
#               a) CalcFratar : Calculates a Fratar/IPF on a seed matrix given row and column (P and A) totals
#               b) CalcSinglyConstrained : Calculates a singly constrained trip distribution for given P/A vectors and a
#                  friction factor matrix
#               c) CalcDoublyConstrained : Calculates a doubly constrained trip distribution for given P/A vectors and a
#                  friction factor matrix (P and A should be balanced before usage, if not then A is scaled to P)
#               d) CalcMultiFratar : Applies fratar model to given set of trip matrices with multiple target production vectors and one attraction vector
#               e) CalcMultiDistribute : Applies gravity model to a given set of frication matrices with multiple production vectors and one target attraction vector
#
#              **All input vectors are expected to be numpy arrays
#
# Author:      Chetan Joshi, Portland OR
# Dependencies:numpy [www.numpy.org]
# Created:     5/14/2015
#
# Copyright:   (c) Chetan Joshi 2015
# Licence:     Permission is hereby granted, free of charge, to any person obtaining a copy
#              of this software and associated documentation files (the "Software"), to deal
#              in the Software without restriction, including without limitation the rights
#              to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#              copies of the Software, and to permit persons to whom the Software is
#              furnished to do so, subject to the following conditions:
#
#              The above copyright notice and this permission notice shall be included in all
#              copies or substantial portions of the Software.
#
#              THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#              IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#              FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#              AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#              LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#              OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#              SOFTWARE.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#
import numpy

def CalcFratar(ProdA, AttrA, Trips1, maxIter = 10, print_balance=False):
    '''Calculates fratar trip distribution
       ProdA = Production target as array
       AttrA = Attraction target as array
       Trips1 = Seed trip table for fratar
       maxIter (optional) = maximum iterations, default is 10
       Returns fratared trip table
    '''
    #print('Checking production, attraction balancing:')
    sumP = sum(ProdA)
    sumA = sum(AttrA)
    #print('Production: ', sumP)
    #print('Attraction: ', sumA)
    if sumP != sumA:
        if print_balance:
            print('Productions and attractions do not balance, attractions will be scaled to productions!')
        AttrA = AttrA*(sumP/sumA)
    else:
        if print_balance:
            print('Production, attraction balancing OK.')
    #Run 2D balancing --->
    for balIter in range(0, maxIter):
        ComputedProductions = Trips1.sum(1)
        ComputedProductions[ComputedProductions==0]=1
        OrigFac = (ProdA/ComputedProductions)
        Trips1 = Trips1*OrigFac[:, numpy.newaxis]

        ComputedAttractions = Trips1.sum(0)
        ComputedAttractions[ComputedAttractions==0]=1
        DestFac = (AttrA/ComputedAttractions)
        Trips1 = Trips1*DestFac
    return Trips1

def CalcSinglyConstrained(ProdA, AttrA, F):
    '''Calculates singly constrained trip distribution for a given friction factor matrix
    ProdA = Production array
    AttrA = Attraction array
    F = Friction factor matrix
    Resutrns trip table
    '''
    SumAjFij = (AttrA*F).sum(1)
    SumAjFij[SumAjFij==0]=0.0001
    return ProdA*(AttrA*F).transpose()/SumAjFij

def CalcDoublyConstrained(ProdA, AttrA, F, maxIter = 10):
    '''Calculates doubly constrained trip distribution for a given friction factor matrix
    ProdA = Production array
    AttrA = Attraction array
    F = Friction factor matrix
    maxIter (optional) = maximum iterations, default is 10
    Returns trip table
    '''
    worse = F.min()
    F = F / worse

    Trips1 = numpy.zeros((len(ProdA),len(ProdA)))
    print('Checking production, attraction balancing:')
    sumP = sum(ProdA)
    sumA = sum(AttrA)
    print('Production: ', sumP)
    print('Attraction: ', sumA)
    if sumP != sumA:
        print('Productions and attractions do not balance, attractions will be scaled to productions!')
        AttrA = AttrA*(sumP/sumA)
        AttrT = AttrA.copy()
        ProdT = ProdA.copy()
    else:
        print('Production, attraction balancing OK.')
        AttrT = AttrA.copy()
        ProdT = ProdA.copy()

    for balIter in range(0, maxIter):
        for i in range(0,len(ProdA)):
            Trips1[i,:] = ProdA[i]*AttrA*F[i,:]/max(0.000001, sum(AttrA * F[i,:]))

        #Run 2D balancing --->
        ComputedAttractions = Trips1.sum(0)
        ComputedAttractions[ComputedAttractions==0]=1
        AttrA = AttrA*(AttrT/ComputedAttractions)

        ComputedProductions = Trips1.sum(1)
        ComputedProductions[ComputedProductions==0]=1
        ProdA = ProdA*(ProdT/ComputedProductions)

    for i in range(0,len(ProdA)):
            Trips1[i,:] = ProdA[i]*AttrA*F[i,:]/max(0.000001, sum(AttrA * F[i,:]))

    return Trips1

def CalcMultiFratar(Prods, Attr, TripMatrices, maxIter=10):
    '''Applies fratar model to given set of trip matrices with target productions and one attraction vector
    Prods = Array of Productions (n production segments)
    AttrAtt = Array of Attraction ( 1 attraction segment)
    TripMatrices = N-Dim array of seed trip matrices corresponding to ProdAtts
     --> (numTripMats, numZones, numZones)
    maxIter = Maximum number of iterations
    version 1.0
    '''
    numZones = len(Attr)
    numTripMats = len(TripMatrices)
    TripMatrices = numpy.zeros((numTripMats,numZones,numZones))

    ProdOp = Prods.copy()
    AttrOp = Attr.copy()

    #Run 2D balancing --->
    for Iter in range(0, maxIter):
        #ComputedAttractions = numpy.ones(numZones)
        ComputedAttractions = TripMatrices.sum(1).sum(0)
        ComputedAttractions[ComputedAttractions==0]=1
        DestFac = Attr/ComputedAttractions

        for k in range(0, len(numTripMats)):
            TripMatrices[k]=TripMatrices[k]*DestFac
            ComputedProductions = TripMatrices[k].sum(1)
            ComputedProductions[ComputedProductions==0]=1
            OrigFac = Prods[:,k]/ComputedProductions #P[i, k1, k2, k3]...
            TripMatrices[k]=TripMatrices[k]*OrigFac[:, numpy.newaxis]

    return TripMatrices

def CalcMultiDistribute(Prods, Attr, FricMatrices, maxIter = 10):
    '''Prods = List of Production Attributes
       Attr  = Attraction Attribute
       FricMatrices = N-Dim array of friction matrices corresponding to ProdAtts --> (numFrictionMats, numZones, numZones)
       maxIter (optional) = Maximum number of balancing iterations, default is 10
       Returns N-Dim array of trip matrices corresponding to each production segment
    '''
    numZones = len(Attr)
    TripMatrices = numpy.zeros(FricMatrices.shape)
    numFricMats = len(FricMatrices)

    ProdOp = Prods.copy()
    AttrOp = Attr.copy()

    for Iter in range(0, maxIter):
        #Distribution --->
        for k in range(0, numFricMats):
            for i in range(0, numZones):
                if ProdOp[i, k] > 0:
                    TripMatrices[k, i, :] = ProdOp[i, k] * AttrOp * FricMatrices[k, i, :] / max(0.000001, sum(AttrOp * FricMatrices[k, i, :]))
        #Balancing --->
        ComputedAttractions = TripMatrices.sum(1).sum(0)
        ComputedAttractions[ComputedAttractions==0]=1
        AttrOp = AttrOp*(Attr/ComputedAttractions)
        for k in range(0, len(fricmatnos)):
            ComputedProductions = TripMatrices[k].sum(1)
            ComputedProductions[ComputedProductions==0]=1
            OrigFac = Prods[:,k]/ComputedProductions
            ProdOp[:,k] = OrigFac*ProdOp[:,k]
    #Final Distribution --->
    for k in range(0, numFricMats):
        for i in range(0, numZones):
            if ProdOp[i, k] > 0:
                TripMatrices[k, i, :] = ProdOp[i, k] * AttrOp * FricMatrices[k, i, :] / max(0.000001, sum(AttrOp * FricMatrices[k, i, :]))

    return TripMatrices

