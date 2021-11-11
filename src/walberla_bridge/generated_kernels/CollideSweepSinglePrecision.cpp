// kernel generated with pystencils v0.3.4+4.g4fecf0c, lbmpy v0.3.4+6.g2faceda,
// lbmpy_walberla/pystencils_walberla from commit
// b17ca5caf00db7d19f86c5f85c6f67fec6c16aff

//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \\file CollideSweepSinglePrecision.cpp
//! \\ingroup lbm
//! \\author lbmpy
//======================================================================================================================

#include <cmath>

#include "CollideSweepSinglePrecision.h"
#include "core/DataTypes.h"
#include "core/Macros.h"

#define FUNC_PREFIX

#if (defined WALBERLA_CXX_COMPILER_IS_GNU) ||                                  \
    (defined WALBERLA_CXX_COMPILER_IS_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#if (defined WALBERLA_CXX_COMPILER_IS_INTEL)
#pragma warning push
#pragma warning(disable : 1599)
#endif

using namespace std;

namespace walberla {
namespace pystencils {

namespace internal_collidesweepsingleprecision {
static FUNC_PREFIX void collidesweepsingleprecision(
    float *RESTRICT const _data_force, float *RESTRICT _data_pdfs,
    int64_t const _size_force_0, int64_t const _size_force_1,
    int64_t const _size_force_2, int64_t const _stride_force_0,
    int64_t const _stride_force_1, int64_t const _stride_force_2,
    int64_t const _stride_force_3, int64_t const _stride_pdfs_0,
    int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2,
    int64_t const _stride_pdfs_3, float omega_bulk, float omega_even,
    float omega_odd, float omega_shear) {
  const float xi_35 = -omega_shear + 2.0f;
  const float xi_36 = xi_35 * 0.5f;
  const float xi_41 = xi_35 * 0.0833333333333333f;
  const float xi_46 = xi_35 * 0.166666666666667f;
  const float xi_56 = xi_35 * 0.25f;
  const float xi_61 = xi_35 * 0.0416666666666667f;
  const float xi_106 = omega_odd * 0.25f;
  const float xi_112 = omega_odd * 0.0833333333333333f;
  const float xi_149 = omega_shear * 0.25f;
  const float xi_172 = omega_odd * 0.0416666666666667f;
  const float xi_174 = omega_odd * 0.125f;
  const int64_t rr_0 = 0.0f;
  const float xi_118 = rr_0 * 0.166666666666667f;
  const float xi_154 = rr_0 * 0.0833333333333333f;
  for (int64_t ctr_2 = 0; ctr_2 < _size_force_2; ctr_2 += 1) {
    float *RESTRICT _data_pdfs_20_311 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 11 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_31 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_316 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 16 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_36 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 6 * _stride_pdfs_3;
    float *RESTRICT _data_force_20_31 =
        _data_force + _stride_force_2 * ctr_2 + _stride_force_3;
    float *RESTRICT _data_pdfs_20_38 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 8 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_310 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 10 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_34 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 4 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_33 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 3 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_315 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 15 * _stride_pdfs_3;
    float *RESTRICT _data_force_20_32 =
        _data_force + _stride_force_2 * ctr_2 + 2 * _stride_force_3;
    float *RESTRICT _data_pdfs_20_312 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 12 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_35 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 5 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_32 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 2 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_317 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 17 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_318 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 18 * _stride_pdfs_3;
    float *RESTRICT _data_force_20_30 = _data_force + _stride_force_2 * ctr_2;
    float *RESTRICT _data_pdfs_20_314 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 14 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_39 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 9 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_313 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 13 * _stride_pdfs_3;
    float *RESTRICT _data_pdfs_20_30 = _data_pdfs + _stride_pdfs_2 * ctr_2;
    float *RESTRICT _data_pdfs_20_37 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 7 * _stride_pdfs_3;
    for (int64_t ctr_1 = 0; ctr_1 < _size_force_1; ctr_1 += 1) {
      float *RESTRICT _data_pdfs_20_311_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_311;
      float *RESTRICT _data_pdfs_20_31_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_31;
      float *RESTRICT _data_pdfs_20_316_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_316;
      float *RESTRICT _data_pdfs_20_36_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_36;
      float *RESTRICT _data_force_20_31_10 =
          _stride_force_1 * ctr_1 + _data_force_20_31;
      float *RESTRICT _data_pdfs_20_38_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_38;
      float *RESTRICT _data_pdfs_20_310_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_310;
      float *RESTRICT _data_pdfs_20_34_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_34;
      float *RESTRICT _data_pdfs_20_33_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_33;
      float *RESTRICT _data_pdfs_20_315_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_315;
      float *RESTRICT _data_force_20_32_10 =
          _stride_force_1 * ctr_1 + _data_force_20_32;
      float *RESTRICT _data_pdfs_20_312_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_312;
      float *RESTRICT _data_pdfs_20_35_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_35;
      float *RESTRICT _data_pdfs_20_32_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_32;
      float *RESTRICT _data_pdfs_20_317_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_317;
      float *RESTRICT _data_pdfs_20_318_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_318;
      float *RESTRICT _data_force_20_30_10 =
          _stride_force_1 * ctr_1 + _data_force_20_30;
      float *RESTRICT _data_pdfs_20_314_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_314;
      float *RESTRICT _data_pdfs_20_39_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_39;
      float *RESTRICT _data_pdfs_20_313_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_313;
      float *RESTRICT _data_pdfs_20_30_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_30;
      float *RESTRICT _data_pdfs_20_37_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_37;
      for (int64_t ctr_0 = 0; ctr_0 < _size_force_0; ctr_0 += 1) {
        const float xi_198 = _data_pdfs_20_311_10[_stride_pdfs_0 * ctr_0];
        const float xi_199 = _data_pdfs_20_31_10[_stride_pdfs_0 * ctr_0];
        const float xi_200 = _data_pdfs_20_316_10[_stride_pdfs_0 * ctr_0];
        const float xi_201 = _data_pdfs_20_36_10[_stride_pdfs_0 * ctr_0];
        const float xi_202 = _data_force_20_31_10[_stride_force_0 * ctr_0];
        const float xi_203 = _data_pdfs_20_38_10[_stride_pdfs_0 * ctr_0];
        const float xi_204 = _data_pdfs_20_310_10[_stride_pdfs_0 * ctr_0];
        const float xi_205 = _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0];
        const float xi_206 = _data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0];
        const float xi_207 = _data_pdfs_20_315_10[_stride_pdfs_0 * ctr_0];
        const float xi_208 = _data_force_20_32_10[_stride_force_0 * ctr_0];
        const float xi_209 = _data_pdfs_20_312_10[_stride_pdfs_0 * ctr_0];
        const float xi_210 = _data_pdfs_20_35_10[_stride_pdfs_0 * ctr_0];
        const float xi_211 = _data_pdfs_20_32_10[_stride_pdfs_0 * ctr_0];
        const float xi_212 = _data_pdfs_20_317_10[_stride_pdfs_0 * ctr_0];
        const float xi_213 = _data_pdfs_20_318_10[_stride_pdfs_0 * ctr_0];
        const float xi_214 = _data_force_20_30_10[_stride_force_0 * ctr_0];
        const float xi_215 = _data_pdfs_20_314_10[_stride_pdfs_0 * ctr_0];
        const float xi_216 = _data_pdfs_20_39_10[_stride_pdfs_0 * ctr_0];
        const float xi_217 = _data_pdfs_20_313_10[_stride_pdfs_0 * ctr_0];
        const float xi_218 = _data_pdfs_20_30_10[_stride_pdfs_0 * ctr_0];
        const float xi_219 = _data_pdfs_20_37_10[_stride_pdfs_0 * ctr_0];
        const float xi_0 = xi_213 + xi_215;
        const float xi_1 = xi_0 + xi_205;
        const float xi_2 = xi_198 + xi_199 + xi_207;
        const float xi_3 = xi_209 + xi_210;
        const float xi_4 = xi_206 + xi_216;
        const float xi_5 = xi_200 + xi_211;
        const float xi_6 = xi_201 + xi_212;
        const float xi_8 = -xi_216;
        const float xi_9 = -xi_219 + xi_8;
        const float xi_10 = -xi_212;
        const float xi_11 = -xi_217;
        const float xi_12 = -xi_206;
        const float xi_13 = xi_10 + xi_11 + xi_12;
        const float xi_14 = -xi_211;
        const float xi_15 = -xi_204;
        const float xi_16 = xi_14 + xi_15;
        const float xi_17 = -xi_200;
        const float xi_18 = -xi_209;
        const float xi_19 = xi_17 + xi_18;
        const float xi_20 = -xi_213;
        const float xi_21 = xi_10 + xi_20;
        const float xi_22 = -xi_207;
        const float xi_23 = -xi_201;
        const float xi_24 = xi_17 + xi_198 + xi_22 + xi_23;
        const float xi_40 = xi_202 * 0.166666666666667f;
        const float xi_48 = xi_214 * 0.166666666666667f;
        const float xi_52 = xi_208 * 0.166666666666667f;
        const float xi_55 = xi_202 * 0.5f;
        const float xi_59 = xi_214 * 0.0833333333333333f;
        const float xi_63 = xi_202 * 0.0833333333333333f;
        const float xi_73 = xi_208 * 0.0833333333333333f;
        const float xi_84 = -xi_218;
        const float xi_85 = xi_201 * 3.0f + xi_210 * 3.0f + xi_84;
        const float xi_86 =
            omega_even *
            (xi_198 * -3.0f + xi_199 * 3.0f + xi_200 * -3.0f + xi_207 * -3.0f +
             xi_209 * -3.0f + xi_211 * 3.0f + xi_85);
        const float xi_87 =
            xi_198 * 2.0f + xi_200 * 2.0f + xi_207 * 2.0f + xi_209 * 2.0f;
        const float xi_88 = xi_205 * 5.0f + xi_206 * 5.0f + xi_87;
        const float xi_89 =
            omega_even *
            (xi_199 * -2.0f + xi_211 * -2.0f + xi_212 * -5.0f + xi_213 * -5.0f +
             xi_215 * -5.0f + xi_217 * -5.0f + xi_85 + xi_88);
        const float xi_92 = -xi_198;
        const float xi_93 = xi_18 + xi_92;
        const float xi_94 = -xi_203;
        const float xi_97 = -xi_215;
        const float xi_98 = xi_11 + xi_15 + xi_21 + xi_97;
        const float xi_100 = xi_217 * 2.0f;
        const float xi_101 = xi_215 * 2.0f;
        const float xi_102 = xi_212 * 2.0f + xi_213 * 2.0f;
        const float xi_103 =
            omega_even *
            (xi_100 + xi_101 + xi_102 + xi_199 * 5.0f + xi_201 * -4.0f +
             xi_203 * -7.0f + xi_204 * -7.0f + xi_210 * -4.0f + xi_211 * 5.0f +
             xi_216 * -7.0f + xi_219 * -7.0f + xi_84 + xi_88);
        const float xi_104 = xi_209 + xi_92;
        const float xi_105 = xi_104 + xi_14 + xi_199 + xi_200 + xi_22;
        const float xi_107 = xi_105 * xi_106;
        const float xi_108 = xi_219 * 2.0f;
        const float xi_109 = xi_204 * 2.0f;
        const float xi_110 = xi_203 * -2.0f + xi_216 * 2.0f;
        const float xi_111 = -xi_108 + xi_109 + xi_110 + xi_14 + xi_19 + xi_2;
        const float xi_113 = xi_111 * xi_112;
        const float xi_114 = -xi_113;
        const float xi_116 = xi_204 + xi_94;
        const float xi_120 = xi_212 + xi_217;
        const float xi_124 = xi_103 * -0.0198412698412698f;
        const float xi_132 = xi_217 + xi_97;
        const float xi_133 = xi_12 + xi_132 + xi_20 + xi_205 + xi_212;
        const float xi_134 = xi_106 * xi_133;
        const float xi_135 = xi_1 + xi_108 - xi_109 + xi_110 + xi_13;
        const float xi_136 = xi_112 * xi_135;
        const float xi_138 = -xi_136;
        const float xi_139 = xi_200 + xi_207;
        const float xi_140 = xi_139 + xi_210 + xi_23 + xi_93;
        const float xi_141 = xi_106 * xi_140;
        const float xi_144 = -xi_100 - xi_101 + xi_102 + xi_24 + xi_3;
        const float xi_145 = xi_112 * xi_144;
        const float xi_146 = -xi_145;
        const float xi_148 = xi_145;
        const float xi_152 = xi_103 * 0.0138888888888889f;
        const float xi_168 = xi_89 * -0.00714285714285714f;
        const float xi_170 = xi_86 * 0.025f;
        const float xi_173 = xi_144 * xi_172;
        const float xi_175 = xi_140 * xi_174;
        const float xi_176 = xi_103 * -0.00396825396825397f;
        const float xi_180 = xi_111 * xi_172;
        const float xi_181 = xi_105 * xi_174;
        const float xi_187 = xi_89 * 0.0178571428571429f;
        const float xi_190 = xi_133 * xi_174;
        const float xi_191 = xi_135 * xi_172;
        const float vel0Term = xi_1 + xi_203 + xi_204;
        const float vel1Term = xi_2 + xi_219;
        const float vel2Term = xi_217 + xi_3;
        const float rho =
            vel0Term + vel1Term + vel2Term + xi_218 + xi_4 + xi_5 + xi_6;
        const float xi_7 = 1 / (rho);
        const float u_0 = xi_7 * (vel0Term + xi_13 + xi_9);
        const float xi_25 = u_0 * xi_214;
        const float xi_26 = xi_25 * 0.333333333333333f;
        const float xi_32 = -xi_26;
        const float xi_90 = rho * (u_0 * u_0);
        const float xi_129 = rho * u_0;
        const float xi_130 = -vel0Term + xi_120 + xi_129 + xi_219 + xi_4;
        const float xi_131 = xi_118 * xi_130;
        const float xi_158 = xi_130 * xi_154;
        const float u_1 = xi_7 * (vel1Term + xi_16 + xi_19 + xi_203 + xi_8);
        const float xi_27 = u_1 * xi_202;
        const float xi_28 = xi_27 * 0.333333333333333f;
        const float xi_33 = -xi_28;
        const float xi_54 = u_1 * 0.5f;
        const float xi_57 = xi_56 * (u_0 * xi_55 + xi_214 * xi_54);
        const float xi_58 = -xi_57;
        const float xi_95 = rho * (u_1 * u_1);
        const float xi_96 = xi_9 + xi_94 + xi_95;
        const float xi_115 = rho * u_1;
        const float xi_117 =
            -vel1Term + xi_115 + xi_116 + xi_209 + xi_216 + xi_5;
        const float xi_119 = xi_117 * xi_118;
        const float xi_150 = xi_149 * (u_0 * xi_115 + xi_116 + xi_219 + xi_8);
        const float xi_155 = xi_117 * xi_154;
        const float xi_156 = xi_155;
        const float xi_157 = xi_113 + xi_156;
        const float xi_166 = -xi_155;
        const float xi_167 = xi_114 + xi_166;
        const float xi_182 = xi_156 - xi_180 + xi_181;
        const float xi_183 = xi_166 + xi_180 - xi_181;
        const float u_2 = xi_7 * (vel2Term + xi_21 + xi_215 + xi_24);
        const float xi_29 = u_2 * xi_208;
        const float xi_30 = xi_29 * 0.333333333333333f;
        const float xi_31 = (-omega_bulk + 2.0f) * (xi_26 + xi_28 + xi_30);
        const float xi_34 = xi_29 * 0.666666666666667f + xi_32 + xi_33;
        const float xi_37 = -xi_30;
        const float xi_38 = xi_27 * 0.666666666666667f + xi_32 + xi_37;
        const float xi_39 = xi_25 * 0.666666666666667f + xi_33 + xi_37;
        const float xi_42 = xi_34 * xi_41;
        const float xi_43 = -xi_42;
        const float xi_44 = xi_39 * xi_41;
        const float xi_45 = -xi_44;
        const float xi_47 = xi_38 * xi_46 + xi_43 + xi_45;
        const float xi_49 = xi_38 * xi_41;
        const float xi_50 = -xi_49;
        const float xi_51 = xi_39 * xi_46 + xi_43 + xi_50;
        const float xi_53 = xi_34 * xi_46 + xi_45 + xi_50;
        const float xi_60 = xi_44 - xi_59;
        const float xi_62 = -xi_34 * xi_61;
        const float xi_64 = xi_31 * 0.125f;
        const float xi_65 = xi_49 + xi_64;
        const float xi_66 = xi_63 + xi_65;
        const float xi_67 = xi_62 + xi_66;
        const float xi_68 = xi_44 + xi_59;
        const float xi_69 = -xi_63 + xi_65;
        const float xi_70 = xi_62 + xi_69;
        const float xi_71 = xi_56 * (u_2 * xi_55 + xi_208 * xi_54);
        const float xi_72 = -xi_39 * xi_61;
        const float xi_74 = xi_42 + xi_73;
        const float xi_75 = xi_72 + xi_74;
        const float xi_76 = -xi_71;
        const float xi_77 = xi_56 * (u_0 * xi_208 * 0.5f + u_2 * xi_214 * 0.5f);
        const float xi_78 = -xi_77;
        const float xi_79 = -xi_38 * xi_61;
        const float xi_80 = xi_64 + xi_74 + xi_79;
        const float xi_81 = xi_42 - xi_73;
        const float xi_82 = xi_72 + xi_81;
        const float xi_83 = xi_64 + xi_79 + xi_81;
        const float xi_91 = rho * (u_2 * u_2);
        const float xi_99 = omega_bulk * (xi_17 + xi_218 + xi_22 + xi_90 +
                                          xi_91 + xi_93 + xi_96 + xi_98);
        const float xi_121 = xi_201 + xi_210 - xi_91;
        const float xi_122 =
            omega_shear * (xi_0 + xi_120 + xi_121 + xi_16 - xi_199 + xi_96);
        const float xi_123 = xi_122 * 0.125f;
        const float xi_125 =
            omega_shear *
            (xi_121 + xi_199 + xi_205 * -2.0f + xi_206 * -2.0f + xi_211 +
             xi_87 + xi_9 + xi_90 * 2.0f + xi_94 - xi_95 + xi_98);
        const float xi_126 = xi_125 * -0.0416666666666667f;
        const float xi_127 = xi_126 + xi_86 * -0.05f;
        const float xi_128 =
            xi_123 + xi_124 + xi_127 + xi_89 * 0.0142857142857143f;
        const float xi_137 = xi_124 + xi_125 * 0.0833333333333333f +
                             xi_89 * -0.0357142857142857f;
        const float xi_142 =
            rho * u_2 - vel2Term + xi_139 + xi_213 + xi_6 + xi_92 + xi_97;
        const float xi_143 = xi_118 * xi_142;
        const float xi_147 = xi_103 * 0.0158730158730159f - xi_123 + xi_127 +
                             xi_89 * -0.0214285714285714f;
        const float xi_151 = xi_122 * 0.0625f;
        const float xi_153 = -xi_150 + xi_151 + xi_152;
        const float xi_159 = xi_99 * 0.0416666666666667f;
        const float xi_160 = xi_125 * 0.0208333333333333f + xi_159;
        const float xi_161 = -xi_158 + xi_160;
        const float xi_162 = xi_138 + xi_161;
        const float xi_163 = xi_150 + xi_151 + xi_152;
        const float xi_164 = xi_158 + xi_160;
        const float xi_165 = xi_136 + xi_164;
        const float xi_169 = xi_149 * (u_2 * xi_115 + xi_104 + xi_17 + xi_207);
        const float xi_171 = xi_126 + xi_159 + xi_168 + xi_169 + xi_170;
        const float xi_177 = xi_142 * xi_154;
        const float xi_178 = xi_176 + xi_177;
        const float xi_179 = -xi_173 + xi_175 + xi_178;
        const float xi_184 = xi_126 + xi_159 + xi_168 - xi_169 + xi_170;
        const float xi_185 = xi_149 * (u_2 * xi_129 + xi_10 + xi_132 + xi_213);
        const float xi_186 = -xi_151;
        const float xi_188 = -xi_185 + xi_186 + xi_187;
        const float xi_189 = xi_148 + xi_178;
        const float xi_192 = xi_161 - xi_190 + xi_191;
        const float xi_193 = xi_185 + xi_186 + xi_187;
        const float xi_194 = xi_164 + xi_190 - xi_191;
        const float xi_195 = xi_176 - xi_177;
        const float xi_196 = xi_173 - xi_175 + xi_195;
        const float xi_197 = xi_146 + xi_195;
        const float forceTerm_0 =
            xi_31 * -1.5f - xi_34 * xi_36 - xi_36 * xi_38 - xi_36 * xi_39;
        const float forceTerm_1 = xi_40 + xi_47;
        const float forceTerm_2 = -xi_40 + xi_47;
        const float forceTerm_3 = -xi_48 + xi_51;
        const float forceTerm_4 = xi_48 + xi_51;
        const float forceTerm_5 = xi_52 + xi_53;
        const float forceTerm_6 = -xi_52 + xi_53;
        const float forceTerm_7 = xi_58 + xi_60 + xi_67;
        const float forceTerm_8 = xi_57 + xi_67 + xi_68;
        const float forceTerm_9 = xi_57 + xi_60 + xi_70;
        const float forceTerm_10 = xi_58 + xi_68 + xi_70;
        const float forceTerm_11 = xi_66 + xi_71 + xi_75;
        const float forceTerm_12 = xi_69 + xi_75 + xi_76;
        const float forceTerm_13 = xi_60 + xi_78 + xi_80;
        const float forceTerm_14 = xi_68 + xi_77 + xi_80;
        const float forceTerm_15 = xi_66 + xi_76 + xi_82;
        const float forceTerm_16 = xi_69 + xi_71 + xi_82;
        const float forceTerm_17 = xi_60 + xi_77 + xi_83;
        const float forceTerm_18 = xi_68 + xi_78 + xi_83;
        _data_pdfs_20_30_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_0 + xi_103 * 0.0238095238095238f + xi_218 + xi_86 * 0.1f +
            xi_89 * 0.0428571428571429f + xi_99 * -0.5f;
        _data_pdfs_20_31_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_1 - xi_107 + xi_114 + xi_119 + xi_128 + xi_199;
        _data_pdfs_20_32_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_2 + xi_107 + xi_113 - xi_119 + xi_128 + xi_211;
        _data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_3 - xi_131 + xi_134 + xi_136 + xi_137 + xi_206;
        _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_4 + xi_131 - xi_134 + xi_137 + xi_138 + xi_205;
        _data_pdfs_20_35_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_5 - xi_141 + xi_143 + xi_146 + xi_147 + xi_210;
        _data_pdfs_20_36_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_6 + xi_141 - xi_143 + xi_147 + xi_148 + xi_201;
        _data_pdfs_20_37_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_7 + xi_153 + xi_157 + xi_162 + xi_219;
        _data_pdfs_20_38_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_8 + xi_157 + xi_163 + xi_165 + xi_203;
        _data_pdfs_20_39_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_9 + xi_162 + xi_163 + xi_167 + xi_216;
        _data_pdfs_20_310_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_10 + xi_153 + xi_165 + xi_167 + xi_204;
        _data_pdfs_20_311_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_11 + xi_171 + xi_179 + xi_182 + xi_198;
        _data_pdfs_20_312_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_12 + xi_179 + xi_183 + xi_184 + xi_209;
        _data_pdfs_20_313_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_13 + xi_188 + xi_189 + xi_192 + xi_217;
        _data_pdfs_20_314_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_14 + xi_189 + xi_193 + xi_194 + xi_215;
        _data_pdfs_20_315_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_15 + xi_182 + xi_184 + xi_196 + xi_207;
        _data_pdfs_20_316_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_16 + xi_171 + xi_183 + xi_196 + xi_200;
        _data_pdfs_20_317_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_17 + xi_192 + xi_193 + xi_197 + xi_212;
        _data_pdfs_20_318_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_18 + xi_188 + xi_194 + xi_197 + xi_213;
      }
    }
  }
}
} // namespace internal_collidesweepsingleprecision

void CollideSweepSinglePrecision::operator()(IBlock *block) {
  auto pdfs = block->getData<field::GhostLayerField<float, 19>>(pdfsID);
  auto force = block->getData<field::GhostLayerField<float, 3>>(forceID);

  auto &omega_odd = this->omega_odd_;
  auto &omega_shear = this->omega_shear_;
  auto &omega_bulk = this->omega_bulk_;
  auto &omega_even = this->omega_even_;
  WALBERLA_ASSERT_GREATER_EQUAL(0, -int_c(force->nrOfGhostLayers()));
  float *RESTRICT const _data_force = force->dataAt(0, 0, 0, 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(0, -int_c(pdfs->nrOfGhostLayers()));
  float *RESTRICT _data_pdfs = pdfs->dataAt(0, 0, 0, 0);
  WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->xSizeWithGhostLayer(),
                                int64_t(cell_idx_c(force->xSize()) + 0));
  const int64_t _size_force_0 = int64_t(cell_idx_c(force->xSize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->ySizeWithGhostLayer(),
                                int64_t(cell_idx_c(force->ySize()) + 0));
  const int64_t _size_force_1 = int64_t(cell_idx_c(force->ySize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->zSizeWithGhostLayer(),
                                int64_t(cell_idx_c(force->zSize()) + 0));
  const int64_t _size_force_2 = int64_t(cell_idx_c(force->zSize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  const int64_t _stride_force_0 = int64_t(force->xStride());
  const int64_t _stride_force_1 = int64_t(force->yStride());
  const int64_t _stride_force_2 = int64_t(force->zStride());
  const int64_t _stride_force_3 = int64_t(1 * int64_t(force->fStride()));
  const int64_t _stride_pdfs_0 = int64_t(pdfs->xStride());
  const int64_t _stride_pdfs_1 = int64_t(pdfs->yStride());
  const int64_t _stride_pdfs_2 = int64_t(pdfs->zStride());
  const int64_t _stride_pdfs_3 = int64_t(1 * int64_t(pdfs->fStride()));
  internal_collidesweepsingleprecision::collidesweepsingleprecision(
      _data_force, _data_pdfs, _size_force_0, _size_force_1, _size_force_2,
      _stride_force_0, _stride_force_1, _stride_force_2, _stride_force_3,
      _stride_pdfs_0, _stride_pdfs_1, _stride_pdfs_2, _stride_pdfs_3,
      omega_bulk, omega_even, omega_odd, omega_shear);
}

void CollideSweepSinglePrecision::runOnCellInterval(
    const shared_ptr<StructuredBlockStorage> &blocks,
    const CellInterval &globalCellInterval, cell_idx_t ghostLayers,
    IBlock *block) {
  CellInterval ci = globalCellInterval;
  CellInterval blockBB = blocks->getBlockCellBB(*block);
  blockBB.expand(ghostLayers);
  ci.intersect(blockBB);
  blocks->transformGlobalToBlockLocalCellInterval(ci, *block);
  if (ci.empty())
    return;

  auto pdfs = block->getData<field::GhostLayerField<float, 19>>(pdfsID);
  auto force = block->getData<field::GhostLayerField<float, 3>>(forceID);

  auto &omega_odd = this->omega_odd_;
  auto &omega_shear = this->omega_shear_;
  auto &omega_bulk = this->omega_bulk_;
  auto &omega_even = this->omega_even_;
  WALBERLA_ASSERT_GREATER_EQUAL(ci.xMin(), -int_c(force->nrOfGhostLayers()));
  WALBERLA_ASSERT_GREATER_EQUAL(ci.yMin(), -int_c(force->nrOfGhostLayers()));
  WALBERLA_ASSERT_GREATER_EQUAL(ci.zMin(), -int_c(force->nrOfGhostLayers()));
  float *RESTRICT const _data_force =
      force->dataAt(ci.xMin(), ci.yMin(), ci.zMin(), 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(ci.xMin(), -int_c(pdfs->nrOfGhostLayers()));
  WALBERLA_ASSERT_GREATER_EQUAL(ci.yMin(), -int_c(pdfs->nrOfGhostLayers()));
  WALBERLA_ASSERT_GREATER_EQUAL(ci.zMin(), -int_c(pdfs->nrOfGhostLayers()));
  float *RESTRICT _data_pdfs = pdfs->dataAt(ci.xMin(), ci.yMin(), ci.zMin(), 0);
  WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->xSizeWithGhostLayer(),
                                int64_t(cell_idx_c(ci.xSize()) + 0));
  const int64_t _size_force_0 = int64_t(cell_idx_c(ci.xSize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->ySizeWithGhostLayer(),
                                int64_t(cell_idx_c(ci.ySize()) + 0));
  const int64_t _size_force_1 = int64_t(cell_idx_c(ci.ySize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->zSizeWithGhostLayer(),
                                int64_t(cell_idx_c(ci.zSize()) + 0));
  const int64_t _size_force_2 = int64_t(cell_idx_c(ci.zSize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  const int64_t _stride_force_0 = int64_t(force->xStride());
  const int64_t _stride_force_1 = int64_t(force->yStride());
  const int64_t _stride_force_2 = int64_t(force->zStride());
  const int64_t _stride_force_3 = int64_t(1 * int64_t(force->fStride()));
  const int64_t _stride_pdfs_0 = int64_t(pdfs->xStride());
  const int64_t _stride_pdfs_1 = int64_t(pdfs->yStride());
  const int64_t _stride_pdfs_2 = int64_t(pdfs->zStride());
  const int64_t _stride_pdfs_3 = int64_t(1 * int64_t(pdfs->fStride()));
  internal_collidesweepsingleprecision::collidesweepsingleprecision(
      _data_force, _data_pdfs, _size_force_0, _size_force_1, _size_force_2,
      _stride_force_0, _stride_force_1, _stride_force_2, _stride_force_3,
      _stride_pdfs_0, _stride_pdfs_1, _stride_pdfs_2, _stride_pdfs_3,
      omega_bulk, omega_even, omega_odd, omega_shear);
}

} // namespace pystencils
} // namespace walberla

#if (defined WALBERLA_CXX_COMPILER_IS_GNU) ||                                  \
    (defined WALBERLA_CXX_COMPILER_IS_CLANG)
#pragma GCC diagnostic pop
#endif

#if (defined WALBERLA_CXX_COMPILER_IS_INTEL)
#pragma warning pop
#endif