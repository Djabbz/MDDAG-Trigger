// Copyright (C) 2003--2004 Ronan Collobert (collober@idiap.ch)
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

#include "ClassNLLCriterion.h"

namespace Torch {

ClassNLLCriterion::ClassNLLCriterion(ClassFormat *class_format_) : Criterion(class_format_->n_classes)
{
  class_format = class_format_;
}

void ClassNLLCriterion::frameForward(int t, real *f_inputs, real *f_outputs)
{
  int the_class = class_format->getClass(data->targets->frames[t]);
  f_outputs[0] = -f_inputs[the_class];
}

void ClassNLLCriterion::frameBackward(int t, real *f_inputs, real *beta_, real *f_outputs, real *alpha_)
{
  if(partial_backprop)
    return;

  int the_class = class_format->getClass(data->targets->frames[t]);
  for(int i = 0; i < n_inputs; i++)
    beta_[i] = 0;
  beta_[the_class] = -1;
}

ClassNLLCriterion::~ClassNLLCriterion()
{
}

}
