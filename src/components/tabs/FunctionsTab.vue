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
        <FileFuncButton
          v-for="(file) of files"
          :header="`${file.filename} (${file.functions.length})`"
          :key="file.filename"
          :selected="selectedFile === file.filename"
          @click="selectFile(file.filename)"
        >
          <div v-if="getFunctions(file.filename).length === 0">(empty)</div>
          <div
            v-else
            v-for="(func, index) of getFunctions(file.filename)"
            :key="index"
          >
            {{ func.signature() }}
          </div>
        </FileFuncButton>
      </div>
    </div>

    <div id="right" class="panel">
<!--      TODO: STYLE LIKE IDE-->
      <div class="button-list">
        <div v-if="selectedFile === null" class="text-center">
          No File Selected
        </div>
        <div
          v-else
          class="pre-formatted"
          v-for="(func) of getFunctions(this.selectedFile)"
          :key="func.name"
          v-text="`${func.toString()}\n`"
        />
      </div>
      <div class="confirm-button">
        <UIButton
          text="Delete"
          @click="deleteFile"
          :disabled="selectedFile === null"
        />
        <UIButton
          text="Edit"
          :primary="true"
          @click="editFile"
          :disabled="selectedFile === null"
        />
      </div>
    </div>

  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import UIButton from '@/components/buttons/UIButton.vue';
import parse from '@/app/parser/parser';
import ParsedFunction from '@/app/parser/ParsedFunction';
import { Result } from '@/app/util';
import { Getter, Mutation } from 'vuex-class';
import { FilenamesList, ParsedFile } from '@/store/codeVault/types';
import { uniqueTextInput } from '@/inputs/prompt';
import FileFuncButton from '@/components/buttons/FileFuncButton.vue';
import { mapGetters } from 'vuex';

@Component({
  components: {
    FileFuncButton,
    Tab,
    Tabs,
    UIButton,
  },
  computed: mapGetters(['files']),
})
export default class FunctionsTab extends Vue {
  @Getter('filenames') filenames!: Set<string>;
  @Getter('file') file!: (filename: string) => ParsedFile;
  @Mutation('addFile') addFile!: (file: ParsedFile) => void;
  @Mutation('deleteFile') delFile!: (filename: string) => void;
  @Mutation('openFile') openFile!: (filename: string) => void;
  @Getter('filenamesList') filenamesList!: FilenamesList;

  private selectedFile: string | null = null;

  private selectFile(filename: string) {
    this.selectedFile = filename;
  }

  private getFunctions(filename: string | null): ParsedFunction[] {
    return filename !== null ? this.file(filename).functions : [];
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
      if (parsed instanceof Error) {
        console.error(parsed);
      } else {
        const file = { filename: files[0].name, functions: parsed, open: false };
        this.addFile(file);
        this.saveToCookies(file);
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

    const file = { filename: `${name}.py`, functions: [], open: true };
    this.addFile(file);
    this.saveToCookies(file);
  }

  private deleteFile() {
    // TODO: Ran through nodes using function and remove nodes
    if (this.selectedFile !== null) {
      const filename = this.selectedFile;
      this.selectedFile = null;
      this.delFile(filename);
    }
  }

  private editFile() {
    if (this.selectedFile !== null) {
      this.openFile(this.selectedFile);
    }
  }

  private saveToCookies(file: ParsedFile) {
    this.$cookies.set('unsaved-code-vault', this.filenamesList);
    this.$cookies.set(`unsaved-file-${file.filename}`, file);
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
    padding-top: 0.5em;
    padding-bottom: 0.5em;
  }

  .button {
    margin-right: 1em;
  }

  .pre-formatted {
    white-space: pre;
  }
</style>
