//
// Created by hitonami on 2021/4/18.
//

#ifndef IR_RULE_H
#define IR_RULE_H

#include <schedule.h>
#include <auto_schedule/sketch.h>

namespace ir {

class Rule {
  public:
    virtual int analyze(Schedule &schedule) = 0;
    virtual SketchPart gen_part(int p) = 0;
};

}

#endif //IR_RULE_H