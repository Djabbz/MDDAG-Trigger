//
//  MDDAGExploration.h
//  MDDAG
//
//  Created by ∂jabbz on 21/10/13.
//  Copyright (c) 2013 AppStat. All rights reserved.
//

#ifndef __MDDAG__MDDAGExploration__
#define __MDDAG__MDDAGExploration__

#include <iostream>


#include "cagentcontroller.h"
#include "cparameters.h"

#include "AdaBoostMDPClassifierContinous.h"

using namespace MultiBoost;

class CAbstractFeatureStochasticEstimatedModel;
class CTransitionFunction;
class CAbstractQFunction;
class CActionSet;
class CFeatureList;
class CActionStatistics;
class CAbstractVFunction;
class CQFunctionFromTransitionFunction;
class CStateCollectionImpl;

//class AdaBoostMDPClassifierContinous;

//using namespace MultiBoost;


#include "newmat/newmat.h"

class MDDAGExploration : public CActionDistribution
{
protected:
public:
    //	double epsilon;
    
	MDDAGExploration(double epsilon, AdaBoostMDPClassifierContinous *classifier, int mode, double factor);
	virtual void getDistribution(CStateCollection *state, CActionSet *availableActions, double *values);
    
protected:
    AdaBoostMDPClassifierContinous *_classifier;
    int _mode;
    double _factor;
};


#endif /* defined(__MDDAG__MDDAGExploration__) */
