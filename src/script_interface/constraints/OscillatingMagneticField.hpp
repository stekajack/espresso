#ifndef SCRIPT_INTERFACE_CONSTRAINTS_OSCILLATINGMAGNETICFIELD_HPP
#define SCRIPT_INTERFACE_CONSTRAINTS_OSCILLATINGMAGNETICFIELD_HPP

#include "core/constraints/Constraint.hpp"
#include "core/constraints/OscillatingMagneticField.hpp"

namespace ScriptInterface {
namespace Constraints {

class OscillatingMagneticField : public Constraint {
public:
  OscillatingMagneticField()
      : m_constraint(new ::Constraints::OscillatingMagneticField()) {
    add_parameters({{"direction",
                     [this](Variant const &v) {
                       m_constraint->set_direction(get_value<Utils::Vector3d>(v));
                     },
                     [this]() { return m_constraint->direction();}
                      },
                    {"magnitude",
                     [this](Variant const &v) {
                       m_constraint->set_magnitude(get_value<double>(v));
                     },
                     [this]() { return m_constraint->magnitude();}
                    },
                    {"frequency",
                     [this](Variant const &v) {
                       m_constraint->set_frequency(get_value<double>(v));
                     },
                     [this]() { return m_constraint->frequency();}
                    },
                     {"phase_shift",
                     [this](Variant const &v) {
                       m_constraint->set_phase_shift(get_value<double>(v));
                     },
                     [this]() { return m_constraint->phase_shift();}
                     }});
  }

  std::shared_ptr<::Constraints::Constraint> constraint() override {
    return std::static_pointer_cast<::Constraints::Constraint>(m_constraint);
  }
  std::shared_ptr<const ::Constraints::Constraint> constraint() const override {
    return std::static_pointer_cast<::Constraints::Constraint>(m_constraint);
  }
  std::shared_ptr<::Constraints::OscillatingMagneticField>
  oscillating_magnetic_field() const {
    return m_constraint;
  }

private:
  /* The actual constraint */
  std::shared_ptr<::Constraints::OscillatingMagneticField> m_constraint;
};

} /* namespace Constraints */
} /* namespace ScriptInterface */

#endif
