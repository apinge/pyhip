cd /opt/pyhip/tests/flydsl

  mkdir -p test_gemm_splitk

  FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
  FLYDSL_DUMP_IR=1 \
  FLYDSL_DUMP_DIR=/opt/pyhip/tests/flydsl/test_gemm_splitk \
  python3 test_gemm.py --alg splitk
