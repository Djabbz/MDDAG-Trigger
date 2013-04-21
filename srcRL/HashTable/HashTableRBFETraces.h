/*
 *  RBFQETraces.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __HASHTABLERBFTRACE_H
#define __HASHTABLERBFTRACE_H

//#define RBFDEB

#include "cqetraces.h"
#include "cstatecollection.h"
#include "HashTableETraces.h" // for MDDAGState
#include "AdaBoostMDPClassifierAdv.h"
//#include "HashTable.h"
//#include <vector>
//#include <list>

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: this structure supposes that we never come back to a state. It's fine for now.
class HashTableRBFETraces: public CAbstractQETraces
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
    
    HashTableRBFETraces(CAbstractQFunction *qFunction, CStateProperties *modelState) : CAbstractQETraces(qFunction) {
//        _learningRate = qFunction->getParameter("QLearningRate");
//        _eTraceStates = new CStateCollectionList(modelState);
    }
    
    // -----------------------------------------------------------------------------------
    
    virtual ~HashTableRBFETraces() { //delete _eTraceStates;
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