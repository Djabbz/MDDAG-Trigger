/*
 *  RBFQETraces.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __HASHTABLETRACE_H
#define __HASHTABLETRACE_H

//#define RBFDEB

#include "cqetraces.h"
#include "cstatecollection.h"

#include "AdaBoostMDPClassifierAdv.h"
//#include "HashTable.h"
//#include <vector>
//#include <list>

using namespace std;

struct MDDAGState {
    
//    MDDAGState() {};
    MDDAGState(CState* state)
    {
        discreteStates.clear();
        continuousStates.clear();
        for (int i = 0; i < state->getNumContinuousStates(); ++i)
            continuousStates.push_back(state->getContinuousState(i));

        for (int i = 0; i < state->getNumDiscreteStates(); ++i)
        {
            discreteStates.push_back(state->getDiscreteState(i));
        }
    }
    
    vector<int> discreteStates;
    vector<AlphaReal> continuousStates;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: this structure supposes that we never come back to a state. It's fine for now.
class HashTableETraces: public CAbstractQETraces
{
protected:
//    vector<RBFParams> _margins;
//	vector<int>	 _iters;
//    CustomETraceMulti _eTraces;
//    int _numDimensions;

	list<CAction*>                  _actions;
//    double                      _learningRate;
//	CStateCollection*           _currentState;
    
//    CStateCollectionList*       _eTraceStates;
//    list<CStateCollection*>         _eTraceStates;
    list<MDDAGState>                _eTraceStates;
    list<double>                    _eTraces;
    
    
public:
    
    // -----------------------------------------------------------------------------------
    
    HashTableETraces(CAbstractQFunction *qFunction, CStateProperties *modelState) : CAbstractQETraces(qFunction) {
//        _learningRate = qFunction->getParameter("QLearningRate");
//        _eTraceStates = new CStateCollectionList(modelState);
    }
    
    // -----------------------------------------------------------------------------------
    
    virtual ~HashTableETraces() { //delete _eTraceStates;
    };
    
    // -----------------------------------------------------------------------------------
    
    virtual void updateETraces(CAction *action, CActionData *data = NULL) ;
    // -----------------------------------------------------------------------------------
    
    virtual void addETrace(CStateCollection *state, CAction *action, double factor = 1.0, CActionData *data = NULL) ;
    
    // -----------------------------------------------------------------------------------
    
    virtual void updateQFunction(double td) ;
    
    // -----------------------------------------------------------------------------------
    
    virtual void resetETraces() ;
    
    // -----------------------------------------------------------------------------------
    
};


#endif