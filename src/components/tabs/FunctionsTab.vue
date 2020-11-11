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
        <div class="ml-3">
          <UIButton text="Create File" @click="createFile"/>
        </div>
      </div>
      <div class="button-list">
        <FileFuncButton header="SelectedFile.py (2)" :selected="true">
          def func(x, y)...
        </FileFuncButton>
      </div>
    </div>
    <div id="right" class="panel">
      SelectedFile.py
      <div class="button-list">
        <FileFuncButton header="def func(x, y):">
          pass...
        </FileFuncButton>
        <FileFuncButton header="def func2(x):">
          pass...
        </FileFuncButton>
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
import { ParsedFile } from '@/store/codeVault/types';
import { uniqueTextInput } from '@/inputs/prompt';
import FileFuncButton from '@/components/buttons/FileFuncButton.vue';

@Component({
  components: {
    FileFuncButton,
    Tab,
    Tabs,
    UIButton,
  },
})
export default class FunctionsTab extends Vue {
  @Getter('filenames') filenames!: Set<string>;
  @Getter('file') file!: (filename: string) => ParsedFile | undefined;
  @Mutation('addFile') addFile!: (file: ParsedFile) => void;

  // Trigger click of input tag for uploading file
  private uploadFile = () => {
    const element = document.getElementById('upload-python-file');
    if (!element) return;
    element.click();
  }

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
      this.filenames, 'Please enter a unique name for the file',
    );
    if (name === null) return;

    this.addFile({ filename: name, functions: [] });
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
  }
  .panel {
    user-select: none;
    padding: 1em;
    border-top: var(--grey) 1px solid;
  }
  .button-list {
    margin-top: 1em;
  }
</style>
