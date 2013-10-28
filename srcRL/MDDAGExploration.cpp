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



MDDAGExploration::MDDAGExploration(double epsilon, AdaBoostMDPClassifierContinous *classifier,  int mode, double factor)
{
	addParameter("EpsilonGreedy", epsilon);
    _classifier = classifier;
    _mode = mode;
    _factor = factor;
}

void MDDAGExploration::getDistribution(CStateCollection *, CActionSet *availableActions, double *actionValues)
{
    int currentClassifier = _classifier->getCurrentClassifier();
    int numIterations = _classifier->getIterNum();
	size_t numValues = availableActions->size();
	

    double epsilon;
    
    if (_mode == 1) {
        epsilon = (getParameter("EpsilonGreedy") * (currentClassifier + 1));
    }
    else if (_mode == 2) {
        epsilon = (pow(getParameter("EpsilonGreedy"), 2) * (currentClassifier + 1)) ; //seems to work the best
    }
    else if (_mode == 3)
    {
        epsilon = (getParameter("EpsilonGreedy") * (currentClassifier + 1)/numIterations) ;
    }
    else if (_mode == 4)
    {
        epsilon = pow(getParameter("EpsilonGreedy"), 2) * (currentClassifier + 1) / _factor ; //cause the factor is powered before the call
    }
    else {
        cout << "Error: wrong adaptive epsilon mode." << endl;
        assert(false);
    }

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