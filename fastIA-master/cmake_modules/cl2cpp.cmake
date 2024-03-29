MESSAGE(STATUS "running cl2cpp")

file(GLOB cl_list "${CL_DIR}/*.cl" )

file(WRITE ${OUTPUT} "// This file is auto-generated. Do not edit!\n\n")

file(WRITE ${OUTPUT} "#include \"opencl/utils/ocl_source_registry.h\"\n\n")

foreach(cl ${cl_list})
  get_filename_component(cl_filename "${cl}" NAME_WE)
  #message("${cl_filename}")

  file(READ "${cl}" lines)

  string(REPLACE "\r" "" lines "${lines}\n")
  string(REPLACE "\t" "  " lines "${lines}")

  string(REGEX REPLACE "/\\*([^*]/|\\*[^/]|[^*/])*\\*/" ""   lines "${lines}") # multiline comments
  string(REGEX REPLACE "/\\*([^\n])*\\*/"               ""   lines "${lines}") # single-line comments
  string(REGEX REPLACE "[ ]*//[^\n]*\n"                 "\n" lines "${lines}") # single-line comments
  string(REGEX REPLACE "\n[ ]*(\n[ ]*)*"                "\n" lines "${lines}") # empty lines & leading whitespace
  string(REGEX REPLACE "^\n"                            ""   lines "${lines}") # leading new line

  string(REPLACE "\\" "\\\\" lines "${lines}")
  string(REPLACE "\"" "\\\"" lines "${lines}")
  string(REPLACE "\n" "\\n\"\n\"" lines "${lines}")

  string(REGEX REPLACE "\"$" "" lines "${lines}") # unneeded " at the eof

  file(APPEND ${OUTPUT} "const char* ${cl_filename}=\"${lines};\n")
  file(APPEND ${OUTPUT} "const int ${cl_filename}_sideefect
        = SourceRegistry::getInstance().registerSource(\"${cl_filename}\", ${cl_filename});\n\n")

endforeach()

#file(APPEND ${OUTPUT} "}\n}\n")
