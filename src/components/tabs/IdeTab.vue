<template>
  <div class="h-100">
    <div id="ace"/>
    <div class="save-banner">
      <div>
        <UIButton text="Cancel" @click="cancel"/>
      </div>
      <div>
        <UIButton text="Save" :primary="true" @click="save"/>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue, Watch } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import Ace from 'brace';
import 'brace/mode/python';
import '@/assets/ivann-theme';
import { Getter, Mutation } from 'vuex-class';
import Custom from '@/nodes/model/custom/Custom';
import UIButton from '@/components/buttons/UIButton.vue';
import parse from '@/app/parser/parser';
import ParsedFunction from '@/app/parser/ParsedFunction';
import { Result } from '@/app/util';

@Component({
  components: {
    UIButton,
    Tab,
    Tabs,
  },
})
export default class IdeTab extends Vue {
  @Getter('nodeTriggeringCodeVault') nodeTriggeringCodeVault?: Custom;
  @Getter('inCodeVault') inCodeVault!: boolean;
  @Mutation('leaveCodeVault') leaveCodeVault!: () => void;
  @Mutation('linkNode') linkNode!: (node?: Custom) => void;
  private editor?: Ace.Editor;

  private parsedFunction?: Result<ParsedFunction>;

  /**
   * Watches the rendering of the CodeVault in order to update code.
   */
  @Watch('inCodeVault')
  private onInCodeVaultChanged(inCodeVault: boolean) {
    if (inCodeVault && this.editor) {
      this.updateCode();
    }
  }

  mounted() {
    this.editor = Ace.edit('ace');
    this.editor.getSession().setMode('ace/mode/python');
    this.editor.setTheme('ace/theme/ivann');
    this.editor.resize(true);
    this.editor.setOptions({
      tabSize: 2,
      useSoftTabs: true,
    });
    this.editor.$blockScrolling = Infinity; // Get rid unnecessary Console info
    this.editor.on('change', this.onEditorChange);

    this.updateCode();
  }

  private save() {
    if (this.nodeTriggeringCodeVault && this.editor) {
      if (!(this.parsedFunction instanceof Error)) {
        this.nodeTriggeringCodeVault.setInlineCode(this.parsedFunction);
        this.leaveCodeVault();
      } else {
        window.alert('Cannot save function with errors.');
      }
    }
  }

  private cancel() {
    this.leaveCodeVault();
    this.linkNode(undefined); // Unlink node.
  }

  /**
   * On Editor Change, parses the code and shows an Error if there is one.
   */
  private onEditorChange() {
    if (this.editor) {
      const code = this.editor.getValue();
      const functionsOrError = parse(code);
      if (!(functionsOrError instanceof Error)) {
        if (functionsOrError.length > 0) {
          this.editor.getSession().clearAnnotations();
          [this.parsedFunction] = functionsOrError;
        }
      } else {
        this.showError(functionsOrError);
        this.parsedFunction = functionsOrError;
      }
    }
  }

  private showError(error: Error) {
    if (this.editor) {
      this.editor.getSession().setAnnotations([{
        row: 0,
        column: 0,
        text: error.message,
        type: 'error',
      }]);
    }
  }
  /**
   * If the CodeVault was entered from a Custom Node, update code.
   * Else, empties the editor.
   */
  private updateCode() {
    if (this.editor) {
      if (this.nodeTriggeringCodeVault) {
        const inlineCode = this.nodeTriggeringCodeVault.getInlineCode();
        if (inlineCode) {
          this.editor.setValue(inlineCode.toString());
        }
      } else {
        this.editor.setValue('');
      }
      this.editor.clearSelection();
    }
  }
}
</script>

<style scoped>
.save-banner {
  border-top: 1px solid var(--grey);
  background: var(--background);
  height: 3em;
  display: flex;
  padding-left: calc(100% - 11em);
}
.button {
  margin-top: 14px;
  margin-right: 1rem;
}
#ace {
  border-top: 1px solid var(--grey);
  height: calc(100% - 3em);
  font-size: 1em;
  font-family: monospace;
  font-weight: lighter;
}
</style>
