// Copyright (C) 2003--2004 Samy Bengio (bengio@idiap.ch)
//                
// This file is part of Torch 3.1.
//
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GRAMMAR_INC
#define GRAMMAR_INC

#include "Object.h"

namespace Torch {

/** This class contains the grammar of accepted sentences for a
    speech recognition experiment such as the one using SimpleDecoderSpeechHMM
    A grammar is a transition table where each node is a word.
    The user is responsible to set the transition table as he wishes.
    (by default, there are no transition!)

    @author Samy Bengio (bengio@idiap.ch)
*/
class Grammar : public Object
{
  public:
    /** the number of words in the grammar (different than the number
        of words in the lexicon, because the same word can appear twice
        in the grammar)
    */
    int n_words;
    /// the index of the words (in the lexicon object)
    int* words;
    /** this vector is used in SimpleDecoderSpeechHMM to keep the state 
        index in the decoding state matrix, corresponding to the given word
    */
    int* start;
    /** the transition matrix. each true transition (i,j) means the
        word whose index is i can be followed by the word whose index is j
    */
    bool** transitions;

    ///
    Grammar(int n_words_);
    virtual ~Grammar();
};


}

#endif
