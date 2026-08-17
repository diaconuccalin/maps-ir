

#ifndef MAPS_OPS
#define MAPS_OPS

include "maps/Dialect/Maps/IR/MapsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class Maps_Op<string mnemonic, list<Trait> traits = []>:
    Op<Maps_Dialect, mnemonic, traits>;


def StageOp : Maps_Op<"stage", [SingleBlock, NoTerminator]>{
    let summary = "A stage operation";
    let description = [{
        A stage operation represents a single stage in a pipeline. 
    }];
    let arguments = (ins
        I64Attr:$stage_id,
        SymbolNameAttr:$sym_name);

    let regions = (region SizedRegion<1>:$body);

    // let hasVerifier = 1;

    let assemblyFormat = [{
        $sym_name attr-dict-with-keyword $body
    }];

}


def TileOp : Maps_Op<"tile", [SingleBlock, NoTerminator]>{
    let summary = "A tile operation";

    let description = [{
        A tile operation represents a single tile in a pipeline. 
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        I64Attr:$tile_id,
        DenseI64ArrayAttr:$coords
    );

    let regions = (region SizedRegion<1>:$body);

    // let hasVerifier = 1;

    let assemblyFormat = [{
        $sym_name attr-dict-with-keyword $body
    }];
}

def InitOp : Maps_Op<"init", [SingleBlock,NoTerminator]>{
    let summary = "An init operation";

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{

        attr-dict-with-keyword $body
    }];
}


def RunOp : Maps_Op<"run", [SingleBlock,NoTerminator]>{
    let summary = "A run operation";

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{

        attr-dict-with-keyword $body
    }];
}


def RecvOp : Maps_Op<"recv"> {
  let summary = "Receive a value from a MAPS channel";

  let description = [{
    The `maps.recv` operation receives a value from a named MAPS
    communication channel. The `src_hartid` attribute names the sending hart,
    while `dst_hartid` names the receiving hart; `-1` denotes L2.

    Example:

    ```mlir
    %value = maps.recv @channel {src_hartid = 0 : i64, dst_hartid = 8 : i64}
      : tensor<64x22xf32>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$channel,
    I64Attr:$src_hartid,
    I64Attr:$dst_hartid
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $channel attr-dict `:` type($result)
  }];
}


def SendOp : Maps_Op<"send"> {
  let summary = "Send a value through a MAPS channel";

  let description = [{
    The `maps.send` operation sends an SSA value through a named MAPS
    communication channel. The `src_hartid` attribute names the sending hart;
    `src_hartid = -1` denotes data sourced from L2.

    Example:

    ```mlir
    maps.send %value, @channel {src_hartid = 0 : i64, dst_hartid = 8 : i64}
      : tensor<64x22xf32>
    ```
  }];

  let arguments = (ins
    AnyType:$value,
    FlatSymbolRefAttr:$channel,
    I64Attr:$src_hartid,
    I64Attr:$dst_hartid
  );

  let assemblyFormat = [{
    $value `,` $channel attr-dict `:` type($value)
  }];
}


def StoreOp : Maps_Op<"store"> {
  let summary = "Store a value for later retrival";

  let description = [{
    

    ```mlir
    
    ```
  }];

  let arguments = (ins
    AnyType:$value,
    FlatSymbolRefAttr:$slot
  );

  let assemblyFormat = [{
    $value `,` $slot attr-dict `:` type($value)
  }];
}


def LoadOp : Maps_Op<"load"> {
  let summary = "Load a value";

  let description = [{
    

    ```mlir
    
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$slot
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $slot attr-dict `:` type($result)
  }];
}

def Conv2DOp : Maps_Op<"conv2d"> {
  let summary = "Execute one tile-local dense NCHW convolution";

  let description = [{
    `maps.conv2d` consumes a clamped tile-local NCHW activation, an OIHW
    weight slice, and an optional output-channel bias slice.  Its padding is
    tile-local: it records the elements missing from the clamped activation
    halo on this tile, not merely the Source Model's global padding.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$dilations,
    DenseI64ArrayAttr:$padding
  );

  let results = (outs
    AnyRankedTensor:$output
  );
}

def KernelOp : Maps_Op<"kernel"> {
  let summary = "Describe one tile-local backend kernel";

  let description = [{
    `maps.kernel` preserves a planned tile-local operation through maps-ir and
    MAGIA descriptor generation before a matching runtime implementation
    exists. `kind` names the handwritten deployment ABI operation and `params`
    contains its operation-specific scalar parameters.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    StrAttr:$kind,
    DenseI64ArrayAttr:$params,
    OptionalAttr<StrAttr>:$device_name,
    OptionalAttr<I64Attr>:$collective_stage_id,
    OptionalAttr<I64Attr>:$collective_layer_index,
    OptionalAttr<I64Attr>:$collective_group_id,
    OptionalAttr<DenseI64ArrayAttr>:$collective_participants
  );

  let results = (outs
    Variadic<AnyRankedTensor>:$outputs
  );
}

#endif // MAPS_OPS
