<template>
  <div class="titlebar row py-2">
    <div class="col text-left">
      <img class="img-fluid titlebar-logo mr-2" src="@/assets/images/nn_logo.png" alt="IVANN"/>
      <span class="text">IVANN</span>
    </div>
    <div class="col text-center">
      <span class="text">
        MNIST-Demo
      </span>
    </div>
    <div class="col text-right">
      <span class="icon-button" @click="codegen">
        <i class="titlebar-icon fas fa-code fa-lg mx-2"/>
      </span>
      <input
        type="file"
        id="upload-file"
        style="display: none"
        @change="load"
      >
      <span class="icon-button" @click="uploadFile">
        <i class="titlebar-icon fas fa-folder-open fa-lg mx-2"/>
      </span>
      <span class="icon-button" @click="save">
        <i class="titlebar-icon fas fa-save fa-lg mx-2"/>
      </span>
      <span class="icon-button" @click="newProject">
        <i class="titlebar-icon fas fa-file fa-lg mx-2"/>
      </span>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Getter, Mutation } from 'vuex-class';
import { EditorModel, EditorModels } from '@/store/editors/types';
import { download, downloadPython } from '@/file/Utils';
import {
  FILENAME,
  Save,
  saveEditor,
  saveEditors,
  SaveWithNames,
} from '@/file/EditorAsJson';
import istateToGraph from '@/app/ir/istateToGraph';
import { ParsedFile } from '@/store/codeVault/types';
import EditorType from '@/EditorType';
import { generateModelCode, generateOverviewCode } from '@/app/codegen/codeGenerator';
import Graph from '@/app/ir/Graph';

@Component
export default class Titlebar extends Vue {
  @Getter('currEditorType') currEditorType!: EditorType;
  @Getter('allEditorModels') editorModels!: EditorModels;
  @Getter('currEditorModel') overviewEditor!: EditorModel;
  @Getter('currEditorModel') currEditor!: EditorModel;
  @Getter('saveWithNames') saveWithNames!: SaveWithNames;
  @Getter('files') files!: ParsedFile[];
  @Mutation('loadEditors') loadEditors!: (save: Save) => void;
  @Mutation('loadFiles') loadFiles!: (files: ParsedFile[]) => void;
  @Mutation('resetState') resetState!: () => void;

  private codegen() {
    let generatedCode = '';
    const { name, state } = saveEditor(this.currEditor);
    const graph = istateToGraph(state);
    if (this.currEditorType === EditorType.OVERVIEW) {
      console.log('generating overview');
      const models = this.editorModels.modelEditors.map((editor) => {
        const { name, state } = saveEditor(editor);
        const graph = istateToGraph(state);
        return [graph, name] as [Graph, string];
      });
      const data = this.editorModels.dataEditors.map((editor) => {
        const { name, state } = saveEditor(editor);
        const graph = istateToGraph(state);
        return [graph, name] as [Graph, string];
      });
      generatedCode = generateOverviewCode(graph, models, data);
    } else if (this.currEditorType === EditorType.MODEL) {
      generatedCode = generateModelCode(graph, name);
    }
    console.log(generatedCode);
    // downloadPython('main', generatedCode);
  }

  // Trigger click of input tag for uploading file
  private uploadFile = () => {
    const element = document.getElementById('upload-file');
    if (!element) return;
    element.click();
  }

  private load() {
    const { files } = document.getElementById('upload-file') as HTMLInputElement;
    if (!files) return;

    const fr: FileReader = new FileReader();
    fr.onload = (event) => {
      if (!event.target) return;

      // Load all editors using parsed file and set cookies
      const parsed = JSON.parse(event.target.result as string);
      this.loadEditors(parsed.editors);
      this.loadFiles(parsed.files);
      this.$cookies.keys().forEach((key) => this.$cookies.remove(key));
      this.$cookies.set('unsaved-project', this.saveWithNames);
      // TODO: FE-65 Set cookies for all editors
    };

    // Trigger the file to be read
    fr.readAsText(files[0]);
  }

  private save() {
    const {
      overviewEditor,
      modelEditors,
      dataEditors,
    } = this.editorModels;

    const editorsSaved: Save = {
      overviewEditor: saveEditor(overviewEditor),
      modelEditors: saveEditors(modelEditors),
      dataEditors: saveEditors(dataEditors),
    };

    download(FILENAME, JSON.stringify({
      editors: editorsSaved,
      files: this.files,
    }));
  }

  private newProject() {
    // TODO FE-55 Implement better Confirm dialog
    if (window.confirm('Are you sure you want to create a new project? '
      + 'All unsaved progress will be lost.')) {
      this.resetState();
      this.$cookies.keys().forEach((key) => this.$cookies.remove(key));
      this.$cookies.set('unsaved-project', this.saveWithNames);
      this.$cookies.set('unsaved-editor-Overview', saveEditor(this.editorModels.overviewEditor));
    }
  }
}
</script>

<style lang="scss" scoped>
  .titlebar {
    height: 2.5rem;
    background-color: var(--background-alt);

    border-bottom: 0.08rem solid var(--grey);
  }

  .titlebar-logo {
    height: 1.2rem;
  }

  .text {
    color: var(--foreground);
  }

  .titlebar-icon {
    color: var(--foreground);
  }

  .icon-button {
    background-color: var(--background-alt);
    width: 5rem;
    height: 5rem;
    margin-left: 0.15rem;
    margin-right: 0.15rem;
    padding: 0.3rem 0.1rem;
    position: relative;
    top: 0;

    &:hover {
      background-color: #2c2c2c;
      cursor: pointer;
    }
  }
  &:hover {
    background-color: #2c2c2c;
    cursor: pointer;
  }
}
</style>
