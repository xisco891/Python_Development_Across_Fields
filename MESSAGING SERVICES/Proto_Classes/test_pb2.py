# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Proto_Files/test.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Proto_Files/test.proto',
  package='proto_test',
  syntax='proto2',
  serialized_pb=_b('\n\x16Proto_Files/test.proto\x12\nproto_test\"#\n\x03\x46oo\x12\x1c\n\x03\x62\x61r\x18\x01 \x01(\x0b\x32\x0f.proto_test.Bar\"\x10\n\x03\x42\x61r\x12\t\n\x01i\x18\x01 \x01(\x05')
)




_FOO = _descriptor.Descriptor(
  name='Foo',
  full_name='proto_test.Foo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='bar', full_name='proto_test.Foo.bar', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=38,
  serialized_end=73,
)


_BAR = _descriptor.Descriptor(
  name='Bar',
  full_name='proto_test.Bar',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='i', full_name='proto_test.Bar.i', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=75,
  serialized_end=91,
)

_FOO.fields_by_name['bar'].message_type = _BAR
DESCRIPTOR.message_types_by_name['Foo'] = _FOO
DESCRIPTOR.message_types_by_name['Bar'] = _BAR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Foo = _reflection.GeneratedProtocolMessageType('Foo', (_message.Message,), dict(
  DESCRIPTOR = _FOO,
  __module__ = 'Proto_Files.test_pb2'
  # @@protoc_insertion_point(class_scope:proto_test.Foo)
  ))
_sym_db.RegisterMessage(Foo)

Bar = _reflection.GeneratedProtocolMessageType('Bar', (_message.Message,), dict(
  DESCRIPTOR = _BAR,
  __module__ = 'Proto_Files.test_pb2'
  # @@protoc_insertion_point(class_scope:proto_test.Bar)
  ))
_sym_db.RegisterMessage(Bar)


# @@protoc_insertion_point(module_scope)
