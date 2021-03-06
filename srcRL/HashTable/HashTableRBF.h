/*
 *  RBFBasedQFunction.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef __HASHTABLERBF_H
#define __HASHTABLERBF_H

#include <map>
#include <vector>
#include <cmath> //for round
#include <algorithm>

#include "Defaults.h"

#include "cqfunction.h"
#include "cqetraces.h"
#include "cgradientfunction.h"
#include "cstatemodifier.h"
#include "HashTableStateModifier.h"
#include "AdaBoostMDPClassifierAdv.h"
//#include "cfeaturefunction.h"
#include <newmat/newmat.h>
#include <newmat/newmatio.h>

#include "HashTableRBFETraces.h"
#include "AdaBoostMDPClassifierContinous.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef vector<double> ValueKey;
typedef map<ValueKey, vector<AlphaReal> > ValueTableType;

typedef MDDAGState StateType; //CStateCollection*

class RBFStateModifier;

class HashTableRBF : public CAbstractQFunction//: public RBFBasedQFunctionBinary
{
protected:
    
    ValueTableType                      _valueTable;
	CActionSet*                         _actions;
	int                                 _numberOfActions;
    AlphaReal                           _learningRate;
    CStateModifier*                     _stateProperties;
    int                                 _numDimensions;
    MultiBoost::AdaBoostMDPClassifierContinous*     _classifier;
        
public:
    
    // -----------------------------------------------------------------------------------
    
    HashTableRBF(CActionSet *actions, CStateModifier* sm,  MultiBoost::AdaBoostMDPClassifierContinous* classifier, int numDim);
    
    // -----------------------------------------------------------------------------------
	
    /**
     * Return the value for a given action and a given score
     */
    double getValue(CStateCollection* state, CAction *action, CActionData *data) ;
    
    // -----------------------------------------------------------------------------------
    
    /**
     * Helper function to return the state value
     * or a default value if the state has never
     * been visited.
     */
    bool getTableValue(int actionIndex, ValueKey& key, AlphaReal& outValue, AlphaReal defaultValue = 0.);
    
    // -----------------------------------------------------------------------------------
    
    /**
     * Build a key for the hash table
     * from the current state
     */
    virtual void getKey(StateType& state, ValueKey& key);
    
    // -----------------------------------------------------------------------------------
    
    void setLearningRate(AlphaReal r) { _learningRate = r; }
    
    // -----------------------------------------------------------------------------------
    
    
    /**
     * Return the max value for a
     * given score
     */
    double getMaxValue(StateType& state) ;
    // -----------------------------------------------------------------------------------
    
    void addTableEntry(double tderror, ValueKey& key, int actionIndex) ;
    // -----------------------------------------------------------------------------------
    
//    void updateValue(CStateCollection *state, CAction *action, double td, CActionData * = NULL);    
    void updateValue(MDDAGState& state, CAction *action, double td, CActionData * = NULL);

    
    // -----------------------------------------------------------------------------------
    
    void saveActionValueTable(FILE* stream, int dim=0) ;
    void saveActionValueTable(string filename) ;
    // -----------------------------------------------------------------------------------
    
    CAbstractQETraces* getStandardETraces() { return new HashTableRBFETraces(this, _stateProperties); } ;
    
    // -----------------------------------------------------------------------------------
    
    //    void getGradient(CStateCollection *state, int action, vector<vector<RBFParams> >& gradient);
    //    void getGradient(RBFParams& margin, int currIter, int action, vector<vector<RBFParams> >& gradient);
    //    void loadQFunction(const string& fileName);

};


#endif