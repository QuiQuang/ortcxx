#ifndef __CINNAMON_MODEL_H__
#define __CINNAMON_MODEL_H__

#include <any>
#include <map>
#include <string>
#include <vector>
#include <future>
#include <optional>
#include <iostream>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace Ort;

namespace cinnamon::model
{


class Model
{
public:
  /**
   * 
   */
  Model(
    const string model,
    const optional<map<string, any>> options = nullopt,
    const optional<map<string, optional<map<string, string>>>> providers = nullopt
  );

  /**
   * Create session options included specified execution provider options.
   * 
   * @param options: 
   * @param providers:  
   */
  static SessionOptions getSessionOptions(
    const optional<map<string, any>> options = nullopt,
    const optional<map<string, optional<map<string, string>>>> providers = nullopt
  );
};

}
#endif