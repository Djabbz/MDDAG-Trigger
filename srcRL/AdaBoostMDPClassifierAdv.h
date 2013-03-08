/*
 *  AdaBoostMDPClassifierAdv.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/10/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADABOOST_MDP_CLASSADV_H
#define __ADABOOST_MDP_CLASSADV_H

//////////////////////////////////////////////////////////////////////
// for multiboost
//////////////////////////////////////////////////////////////////////
#include "WeakLearners/BaseLearner.h"
#include "IO/InputData.h"
#include "Utils/Utils.h"
#include "IO/Serialization.h"
#include "IO/OutputInfo.h"
#include "Classifiers/AdaBoostMHClassifier.h"
#include "Classifiers/ExampleResults.h"
#include "bitset"

//////////////////////////////////////////////////////////////////////
// for RL toolbox
//////////////////////////////////////////////////////////////////////
#include "cenvironmentmodel.h"
#include "crewardfunction.h"
#include "caction.h"
#include "cdiscretizer.h"
//////////////////////////////////////////////////////////////////////
// general includes
//////////////////////////////////////////////////////////////////////


using namespace std;

#define MAX_NUM_OF_ITERATION 10000
typedef vector<bitset<MAX_NUM_OF_ITERATION> >	vBitSet;
typedef vector<bitset<MAX_NUM_OF_ITERATION> >*	pVBitSet;

typedef vector<vector<char> >	vVecChar;
typedef vector<vector<char> >*	pVVecChar;


namespace MultiBoost {
	
	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////
	
	// Forward declarations.
	class ExampleResults;
	class InputData;
	class BaseLearner;
	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////
	

	class DataReader {
	public:
		// constructor
		DataReader(const nor_utils::Args& args, int verbose);
		
		// upload the data
		void loadInputData(const string& dataFileName, const string& testDataFileName, const string& testDataFileName2, const string& shypFileName);

		// update example result, and return the alpha of the weak classifier used
		double classifyKthWeakLearner( const int wHypInd, const int instance, ExampleResults* exampleResult );
		
		bool currentClassifyingResult( const int currentIstance, ExampleResults* exampleResult );
        
		double getExponentialLoss( const int currentIstance, ExampleResults* exampleResult );
        double getLogisticLoss( const int currentIstance, ExampleResults* exampleResult );
        
		bool hasithLabel( int currentIstance, int classIdx );
		
		// getter setters
		int getClassNumber() const { return _pCurrentData->getNumClasses(); }
		int getIterationNumber() const { return _numIterations; }
		
		int getNumExamples() const { return _pCurrentData->getNumExamples(); }
		int getTrainNumExamples() const { return _pTrainData->getNumExamples(); }
		int getTestNumExamples() const { return _pTestData->getNumExamples(); }				
		
        int getNumAttributes() const { return _pCurrentData->getNumAttributes();}
        const NameMap& getAttributeNameMap() { return _pCurrentData->getAttributeNameMap();}
        
        set<int> getUsedColumns(int weakhypIdx) { return _weakHypotheses[weakhypIdx]->getUsedColumns();}
        
		void setCurrentDataToTrain() {
			_pCurrentData = _pTrainData; 
			if (_isDataStorageMatrix) _pCurrentMatrix = &_weakHypothesesMatrices[_pCurrentData];			
		}
		void setCurrentDataToTest() { 
			_pCurrentData = _pTestData; 
			if (_isDataStorageMatrix) _pCurrentMatrix = &_weakHypothesesMatrices[_pCurrentData];
		}		

        bool setCurrentDataToTest2() { 
            if (_pTestData2) {
                _pCurrentData = _pTestData2; 
                return true;
            }
            return false;
			
//			if (_isDataStorageMatrix) _pCurrentMatrix = &_weakHypothesesMatrices[_pCurrentData];
		}		
        
		double getAccuracyOnCurrentDataSet();
		
		double getSumOfAlphas() const { return _sumAlphas; }
        
        inline const NameMap& getClassMap()
		{ return _pCurrentData->getClassMap(); }

	protected:
		void calculateHypothesesMatrix();
		
		int						_verbose;		
		double					_sumAlphas;
		
		const nor_utils::Args&  _args;  //!< The arguments defined by the user.		
		int						_currentInstance;
		vector<BaseLearner*>	_weakHypotheses;		
		
		InputData*				_pCurrentData;
		InputData*				_pTrainData;
		InputData*				_pTestData;
		
        InputData*				_pTestData2;
        
		int						_numIterations;	
		
//		map< InputData*, vBitSet > _weakHypothesesMatrices;
//		pVBitSet				   _pCurrentBitset;
		map< InputData*, vVecChar > _weakHypothesesMatrices;
		pVVecChar				   _pCurrentMatrix;
		
		
		bool					_isDataStorageMatrix;
		vector< vector< AlphaReal > > _vs;
		vector< AlphaReal >			_alphas;
	};

	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	class CAdaBoostAction : public CPrimitiveAction
	{
	protected:
		int _mode; // 0 skip, 1 classify, 2 jump to the end
	public:
		CAdaBoostAction( int mode ) : CPrimitiveAction() 
		{
			_mode=mode;
		}
		int getMode() { return _mode;}
	};
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	class AdaBoostDiscreteState : public CAbstractStateDiscretizer
	{
	protected:
		unsigned int _iterNum, _classNum;
		
	public:
		AdaBoostDiscreteState(unsigned int iterNum, unsigned int classNum);
		virtual ~AdaBoostDiscreteState() {};
		
		virtual unsigned int getDiscreteStateNumber(CStateCollection *state);		
	};
	
	
} // end of namespace MultiBoost

#endif // __ADABOOST_MDP_CLASSIFIER_ADV_H

