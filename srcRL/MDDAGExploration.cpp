//
//  MDDAGExploration.cpp
//  MDDAG
//
//  Created by ∂jabbz on 21/10/13.
//  Copyright (c) 2013 AppStat. All rights reserved.
//

//#include "ril_debug.h"
#include "cpolicies.h"
//#include "cutility.h"
//#include "ctheoreticalmodel.h"
//#include "cqfunction.h"
//#include "cvfunction.h"
//#include "cactionstatistics.h"
//#include "cstateproperties.h"
#include "cstate.h"
//#include "cstatecollection.h"
//#include "ctransitionfunction.h"
//#include "cfeaturefunction.h"

#include <assert.h>
#include <math.h>

#include "MDDAGExploration.h"



MDDAGExploration::MDDAGExploration(double epsilon, AdaBoostMDPClassifierContinous *classifier)
{
	addParameter("EpsilonGreedy", epsilon);
    _classifier = classifier;
}

void MDDAGExploration::getDistribution(CStateCollection *, CActionSet *availableActions, double *actionValues)
{
    int currentClassifier = _classifier->getCurrentClassifier();
    int numIterations = _classifier->getIterNum();
	size_t numValues = availableActions->size();
	
//    double epsilon = getParameter("EpsilonGreedy") * (currentClassifier + 1)/numIterations ;
//    double epsilon = getParameter("EpsilonGreedy") * (currentClassifier + 1) ;
    double epsilon = pow(2., getParameter("EpsilonGreedy")) * (currentClassifier + 1) ;
    if (epsilon > 1) epsilon = 1;
    
	double prop = epsilon / numValues;
	double max = actionValues[0];
	int maxIndex = 0;
	
	for (unsigned int i = 0; i < numValues; i++)
	{
		if (actionValues[i] > max)
		{
			max = actionValues[i];
			maxIndex = i;
		}
		actionValues[i] = prop;
	}
	actionValues[maxIndex] += 1 - epsilon;
}