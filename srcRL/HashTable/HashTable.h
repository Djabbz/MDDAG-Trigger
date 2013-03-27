/*
 *  RBFBasedQFunction.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef __HASHTABLE_H
#define __HASHTABLE_H

#include <map>
#include <vector>

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

#include "HashTableETraces.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef vector<AlphaReal> ValueKey;
typedef map<ValueKey, AlphaReal> ValueTableType;

class RBFStateModifier;

class HashTable : public CAbstractQFunction//: public RBFBasedQFunctionBinary
{
protected:
    
    vector<ValueTableType>          _valueTable;
	CActionSet*                     _actions;

	int                             _numberOfActions;
    int                             _numberOfIterations;
    
    int                             _numDimensions;
    
    AlphaReal                       _learningRate;
    
    RBFStateModifier*               _stateProperties;
public:
    
    // -----------------------------------------------------------------------------------
    
    HashTable(CActionSet *actions, CStateModifier* sm ) : CAbstractQFunction(actions)
    {
        _stateProperties = dynamic_cast<RBFStateModifier *>(sm);
        
        const int iterationNumber = _stateProperties->getNumOfIterations();
        const int numOfClasses = _stateProperties->getNumOfClasses();
        
        _numberOfIterations = iterationNumber;
        _numDimensions = numOfClasses;
        
        _actions = actions;
        _numberOfActions = actions->size();
        
        _valueTable.resize(_numberOfActions);
        addParameter("QLearningRate", 0.2);

        
    }

    // -----------------------------------------------------------------------------------
	
    /**
     * Return the value for a given action and a given score
     */
    double getValue(CStateCollection *state, CAction *action, CActionData *data) {
        int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();        
        ValueKey key;
        getKey(state, key);
        return getValue(actionIndex, key);
    };
    
    // -----------------------------------------------------------------------------------
    
    /**
     * Helper function to return the state value
     * or a default value if the state has never
     * been visited.
     */
    double getValue(int actionIndex, ValueKey& key, AlphaReal defaultValue = 0.)
    {
        ValueTableType::const_iterator it = _valueTable[actionIndex].find(key);
        if (it == _valueTable[actionIndex].end()) {
            return defaultValue;
        }
        else {
            return it->second;
        }
    }
    
    // -----------------------------------------------------------------------------------
    
    /**
     * Build a key for the hash table
     * from the current state
     */
    void getKey(CStateCollection *state, ValueKey& key)
    {
        CState* currState = state->getState();
        key.resize(_numDimensions);
        for (int i = 0; i < _numDimensions; ++i) {
            key[i] = currState->getContinuousState(i);
        }
    }
    
    // -----------------------------------------------------------------------------------
    
    void setLearningRate(AlphaReal r) { _learningRate = r; }
    
    // -----------------------------------------------------------------------------------
    
    
    /**
     * Return the max value for a
     * given score
     */
    double getMaxValue(CStateCollection *state)
    {
        ValueKey key;
        getKey(state, key);
        
        AlphaReal maxVal = 0.0;
        
        for( int i=0; i < _numberOfActions; ++i )
        {
            AlphaReal value = getValue(i, key);
            if (value > maxVal) {
                maxVal = value;
            }
        }
        return maxVal;
        cout << endl << "max activation " << maxVal << endl;
    }
    
    // -----------------------------------------------------------------------------------
    
    void addTableEntry(double tderror, CStateCollection *state, int actionIndex)
    {
        ValueKey key;
        getKey(state, key);
        
        ValueTableType::const_iterator it = _valueTable[actionIndex].find(key);
        assert(it == _valueTable[actionIndex].end());

        _valueTable[actionIndex][key] = tderror;
    }
    
    // -----------------------------------------------------------------------------------
    
    void updateValue(CStateCollection *state, CAction *action, double td, CActionData * = NULL)
    {
        if (td != td) {
            assert(false);
        }

        int actionIndex = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
        
        ValueKey key;
        getKey(state, key);
        
        _valueTable[actionIndex][key] += td; //* _learningRate;
    };
    
    // -----------------------------------------------------------------------------------
    
    void saveActionValueTable(FILE* stream, int dim=0)
    {
        fprintf(stream, "Q-Hash Table\n");

        ValueTableType::iterator tableIt = _valueTable.begin();
        
        for (; tableIt != _valueTable.end(); ++tableIt) {
            
            ValueKey& key = (*tableIt)->first;
            AlphaReal value = (*tableIt)->second;
            
            fprintf(stream, "( ");
            for (ValueKey::iterator keyIt = key.begin(); keyIt != key.end(); ++keyIt) {
                fprintf(stream, "%f ", *keyIt);
            }
            fprintf(stream, ")\t");
            
            for (int k = 0; k < _numberOfActions; ++k) {
                fprintf(stream,"classifier %d action %d: ", j,k);
                for (int i = 0; i < _rbfs[k][j].size(); ++i) {
                    fprintf(stream,"%f %f %f ", _rbfs[k][j][i].getAlpha()[dim], _rbfs[k][j][i].getMean()[dim], _rbfs[k][j][i].getSigma()[dim]);
                }
                fprintf(stream, "\n");
            }
            
        }

    }

    // -----------------------------------------------------------------------------------
    
    CAbstractQETraces* getStandardETraces() { return new HashTableETraces(this, _stateProperties); } ;
    
    // -----------------------------------------------------------------------------------
    
    //    void getGradient(CStateCollection *state, int action, vector<vector<RBFParams> >& gradient);
    //    void getGradient(RBFParams& margin, int currIter, int action, vector<vector<RBFParams> >& gradient);
    //    void loadQFunction(const string& fileName);

};

#endif