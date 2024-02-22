# Developer Notes

## Pinocchio

Cheat sheet: https://github.com/fabinsch/pinocchio/blob/main/doc/pinocchio_cheat_sheet.pdf

> What are `oMi`, `oMf` and other terms in `data`?

See <https://github.com/stack-of-tasks/pinocchio/blob/0caf0ca4d07e63834cdc420c703993662c59e01b/include/pinocchio/bindings/python/multibody/data.hpp>.

For example:

- `oMf` is "frames absolute placement (wrt world)"

> What is the `actInv` method?

`actInv` is the inverse of "this" times "other". See <https://github.com/stack-of-tasks/pinocchio/blob/0caf0ca4d07e63834cdc420c703993662c59e01b/include/pinocchio/bindings/python/spatial/se3.hpp#L114>.
