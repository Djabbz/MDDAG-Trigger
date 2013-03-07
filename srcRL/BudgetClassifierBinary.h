//
//  BudgetClassifierBinary.h
//  MDDAG
//
//  Created by ∂jabbz on 04/03/13.
//  Copyright (c) 2013 AppStat. All rights reserved.
//

#ifndef __MDDAG__BudgetClassifierBinary__
#define __MDDAG__BudgetClassifierBinary__

#include <iostream>
#include <fstream>

#include "NameMap.h"
#include "AdaBoostMDPClassifierContinousBinary.h"

using namespace std;

namespace MultiBoost {
    /**
     * AdaBoostMH on budgeted classification
     * \date 04/03/2013
     */
    class BudgetClassifierBinary : public AdaBoostMDPClassifierContinousBinary {
        
    public:
        BudgetClassifierBinary( const nor_utils::Args& args, int verbose, DataReader* datareader, string featureCostFile);
        
        virtual ~BudgetClassifierBinary() {}
        
        double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
		
		///fetches the internal state and stores it in the state object
		virtual void getState(CState *state); ///resets the model
        void doNextState(CPrimitiveAction *act);
        void doResetModel();
        void outPutStatistic( BinaryResultStruct& bres );
        double getClassificationCost();

    protected:
        vector<AlphaReal> _featureCosts;
        vector<bool> _featuresEvaluated;
        
    private:
        BudgetClassifierBinary& operator=( const BudgetClassifierBinary& ) {return *this;}
    };

}
#endif /* defined(__MDDAG__BudgetClassifierBinary__) */
