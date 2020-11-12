<template>
  <div class="d-flex h-100">
    <div id="left" class="panel">
      <div class="d-flex">
        <div>
          <UIButton text="Upload File" @click="uploadFile"/>
          <input
            type="file"
            id="upload-python-file"
            style="display: none"
            @change="load"
          >
        </div>
        <div>
          <UIButton text="Create File" @click="createFile"/>
        </div>
      </div>
      <div class="button-list">
        <FileFuncButton v-for="(file, index) of files"
                        :header="`${file.filename} (${file.functions.length})`"
                        :key="index"
                        :selected="selectedFile === index"
                        @click="selectFile(index)">
          <div v-if="getFunctions(index).length === 0">(empty)</div>
          <div v-else>
            <div v-for="(func, index) of getFunctions(index)" :key="index">
              {{ func.signature() }}
            </div>
          </div>
        </FileFuncButton>
      </div>
    </div>
    <div id="right" class="panel">
      <div class="button-list">
        <div v-if="getFunctions(selectedFile).length === 0" class="text-center">
          {{ selectedFile === -1 ? 'Select a file.' : 'No functions defined!' }}
        </div>
        <FileFuncButton v-for="(func, index) of getFunctions(selectedFile)"
                        :header="`def ${func.name}`"
                        :key="index"
                        :selected="selectedFunction === index"
                        @click="selectFunction(index)">
          {{
            func.toString()
              .slice(0, 100) + (func.toString().length > 100 ? '...' : '')
          }}
        </FileFuncButton>
      </div>
      <div class="confirm-button">
        <UIButton text="Cancel" @click="cancelClick"/>
        <UIButton text="Confirm" :primary="true" @click="confirmClick"/>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue, Watch } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import UIButton from '@/components/buttons/UIButton.vue';
import parse from '@/app/parser/parser';
import ParsedFunction from '@/app/parser/ParsedFunction';
import { Result } from '@/app/util';
import { Getter, Mutation } from 'vuex-class';
import { ParsedFile } from '@/store/codeVault/types';
import { uniqueTextInput } from '@/inputs/prompt';
import FileFuncButton from '@/components/buttons/FileFuncButton.vue';
import Custom from '@/nodes/common/Custom';

@Component({
  components: {
    FileFuncButton,
    Tab,
    Tabs,
    UIButton,
  },
})
export default class FunctionsTab extends Vue {
  @Getter('nodeTriggeringCodeVault') nodeTriggeringCodeVault?: Custom;
  @Getter('inCodeVault') inCodeVault!: boolean;
  @Mutation('leaveCodeVault') leaveCodeVault!: () => void;
  @Getter('filenames') filenames!: Set<string>;
  @Getter('file') file!: (filename: string) => ParsedFile | undefined;
  @Getter('files') files!: ParsedFile[];
  @Mutation('addFile') addFile!: (file: ParsedFile) => void;
  @Getter('fileIndexFromFilename') fileIndexFromFilename!: (filename: string) => number;
  @Getter('functionIndexFromFunctionName') functionIndexFromFunctionName!:
    (fileIndex: number, functionName: string) => number;

  private selectedFile = -1;
  private selectedFunction = -1;

  /**
   * Watches the rendering of the CodeVault in order to update selected function.
   */
  @Watch('inCodeVault')
  private onInCodeVaultChanged(inCodeVault: boolean) {
    if (inCodeVault) {
      if (this.nodeTriggeringCodeVault) {
        const file = this.nodeTriggeringCodeVault.getParsedFileName();
        const func = this.nodeTriggeringCodeVault.getParsedFunction();
        if (file && func) {
          this.selectedFile = this.fileIndexFromFilename(file);
          this.selectedFunction = this.functionIndexFromFunctionName(
            this.selectedFile,
            func.name,
          );
          return;
        }
      }
      this.selectedFile = -1;
      this.selectedFunction = -1;
    }
  }

  private selectFile(index: number) {
    this.selectedFile = index;
    this.selectedFunction = -1;
  }

  private selectFunction(index: number) {
    if (index === this.selectedFunction) {
      this.selectedFunction = -1;
    } else {
      this.selectedFunction = index;
    }
  }

  private getFunctions(index: number): ParsedFunction[] {
    const fileList = this.files;
    if (index >= 0 && index < fileList.length) return fileList[index].functions;
    return [];
  }

  // Trigger click of input tag for uploading file
  private uploadFile = () => {
    const element = document.getElementById('upload-python-file');
    if (!element) return;
    element.click();
  };

  private load() {
    const { files } = document.getElementById('upload-python-file') as HTMLInputElement;
    if (!files) return;

    const fr: FileReader = new FileReader();
    fr.onload = (event) => {
      if (!event.target) return;

      // If filename is not unique, report error and cancel
      const filename: string = files[0].name;
      if (this.file(filename) !== undefined) {
        console.error('Attempted to upload file with non-unique filename');
        return;
      }

      // Parse file - report any errors
      const parsed: Result<ParsedFunction[]> = parse(event.target.result as string);
      if ('message' in parsed) {
        console.error(parsed);
      } else {
        this.addFile({ filename: files[0].name, functions: parsed });
      }
    };

    // Trigger the file to be read
    fr.readAsText(files[0]);
  }

  private createFile() {
    const name: string | null = uniqueTextInput(
      this.filenames, 'Please enter a unique name for the file', '.py',
    );
    if (name === null) return;

    this.addFile({ filename: `${name}.py`, functions: [] });
  }

  private confirmClick() {
    if (this.nodeTriggeringCodeVault) {
      if (this.selectedFile !== -1 && this.selectedFunction !== -1) {
        this.nodeTriggeringCodeVault.setParsedFileName(this.files[this.selectedFile].filename);
        this.nodeTriggeringCodeVault.setParsedFunction(
          this.files[this.selectedFile].functions[this.selectedFunction],
        );
        this.leaveCodeVault();
      } else {
        console.log('Make a selection!');
      }
    }
  }

  private cancelClick() {
    this.leaveCodeVault();
  }
}
</script>

<style scoped>
  #left {
    width: 40%;
    border-right: var(--grey) 1px solid;
  }

  #right {
    width: 60%;
    padding-top: 0;
  }

  .panel {
    user-select: none;
    padding: 1em;
    border-top: var(--grey) 1px solid;
  }

  .button-list {
    margin-top: 1em;
    margin-bottom: 1em;
    height: calc(100% - 3.5em);
  }

  .confirm-button {
    display: flex;
    float: right;
    padding-bottom: 0.5em;
  }

  .button {
    margin-right: 1em;
  }
</style>
