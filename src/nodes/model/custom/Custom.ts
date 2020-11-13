// import { Node } from '@baklavajs/core';
// import { CommonNodes } from '@/nodes/common/Types';
// import parse from '@/app/parser/parser';
//
// export enum CustomOptions{
//   InlineCode = 'Inline Code',
//   Code = 'Code'
// }
// export default class Custom extends Node {
//   type = CommonNodes.Custom;
//   name = CommonNodes.Custom;
//
//   private inputNames: string[] = [];
//
//   private code = '';
//
//   constructor() {
//     super();
//     // TODO: IS THIS STILL NEEDED?
//     // this.addOption(CustomOptions.InlineCode, 'TextAreaOption',
//     //   { text: this.code, hasError: false });
//     this.addOutputInterface('Output');
//     this.events.update.addListener(this, (event: any) => {
//       this.nodeUpdated(event);
//     });
//   }
//
//   private nodeUpdated(event: any) {
//     if (event.name === CustomOptions.InlineCode) {
//       const code = event.option.value.text;
//
//       /* If the code did not change (probably the hasError did), do not parse it */
//       if (code === this.code) {
//         return;
//       }
//
//       this.code = code;
//
//       if (code === '') {
//         this.removeAllInputs();
//       } else {
//         const functions = parse(code);
//
//         if (functions instanceof Error) {
//           this.setError(true);
//         } else {
//           this.setError(false);
//           if (functions.length > 0) {
//             const func = functions[0];
//             this.setInputs(func.args);
//           }
//         }
//       }
//     }
//   }
//
//   private setInputs(inputNames: string[]) {
//     this.removeAllInputs();
//     inputNames.forEach((inputName: string) => {
//       this.addInputInterface(inputName);
//     });
//     this.inputNames = inputNames;
//   }
//
//   private removeAllInputs() {
//     this.inputNames.forEach((inputName: string) => {
//       this.removeInterface(inputName);
//     });
//     this.inputNames = [];
//   }
//
//   private setError(error: boolean) {
//     this.setOptionValue(
//       CustomOptions.InlineCode,
//       { text: this.getOptionValue(CustomOptions.InlineCode).text, hasError: error },
//     );
//   }
// }
